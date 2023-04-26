"""
Generic eval that uses a prompt + classification.
"""
import logging
from collections import Counter
from random import Random
from typing import Optional, Union

import openai

import evals
import evals.record
from evals import CompletionFn
from evals.elsuite.modelgraded.base import ModelGradedSpec
from evals.elsuite.modelgraded.classify_utils import CHOICE_KEY, INVALID_STR, concat_n_completions
from evals.elsuite.utils import PromptFn, scrub_formatting_from_prompt
from evals.registry import Registry


class ModelBasedClassify(evals.Eval):
    invalid_request_during_completion = 0
    invalid_request_during_evaluation = 0

    def __init__(
        self,
        completion_fns: list[CompletionFn],
        samples_jsonl: str,
        modelgraded_spec: str,
        registry: Registry,
        *args,
        max_tokens: int = 1024,
        multicomp_n: Union[int, str] = 1,
        multicomp_temperature: float = 0.4,
        samples_renamings: Optional[dict[str, str]] = None,
        eval_type: Optional[str] = None,
        metaeval: bool = False,
        modelgraded_spec_args: Optional[dict[str, dict[str, str]]] = None,
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        # treat last completion_fn as eval_completion_fn
        self.eval_completion_fn = self.completion_fns[-1]
        if len(self.completion_fns) > 1:
            self.completion_fns = self.completion_fns[:-1]
        n_models = len(self.completion_fns)
        self.max_tokens = max_tokens
        self.samples_jsonl = samples_jsonl
        self.metaeval = metaeval
        self.registry = registry
        if multicomp_n == "from_models":
            assert n_models > 1
            self.multicomp_n = n_models
        else:
            assert isinstance(multicomp_n, int)
            self.multicomp_n = multicomp_n
        if len(self.completion_fns) > 1:
            assert self.multicomp_n == n_models
        self.multicomp_temperature = multicomp_temperature
        self.samples_renamings = samples_renamings or {}

        spec_kwargs = {"multicomp_n": self.multicomp_n}
        self.mg: ModelGradedSpec = self.registry.get_modelgraded_spec(
            modelgraded_spec, **spec_kwargs
        )
        if eval_type:
            self.mg.append_answer_prompt(eval_type)
        if modelgraded_spec_args:
            self.mg.fill_args(**modelgraded_spec_args)

    def eval_sample(self, test_sample: dict, rng: Random) -> None:
        """Evaluate a single sample.

        Recorded metrics are always: one of the self.choice_strings, or "__invalid__".
        """
        # process test_sample
        if self.samples_renamings:
            test_sample = {self.samples_renamings.get(k, k): v for k, v in test_sample.items()}
        if self.multicomp_n > 1:
            test_sample["n"] = self.multicomp_n
        for k in self.mg.input_outputs:
            test_sample[k] = scrub_formatting_from_prompt(test_sample[k])

        # run policy completions
        completions = {}
        try:
            for k, v in self.mg.input_outputs.items():
                if v in test_sample:  # test_sample already has completion, skip.
                    continue
                if self.multicomp_n > 1 and v in self.mg.completion_sample_templates:
                    completion_i_s = []
                    for i in range(self.multicomp_n):
                        if len(self.completion_fns) > 1:
                            # use a separate model for each completion
                            completion_fn = self.completion_fns[i]
                        else:
                            # use the single model for all completions
                            completion_fn = self.completion_fn
                        get_input_completion = PromptFn(
                            test_sample[k],
                            completion_fn=completion_fn,
                            max_tokens=self.max_tokens,
                            temperature=self.multicomp_temperature,
                        )
                        completion_i, _ = get_input_completion()
                        completion_i_s.append(completion_i)
                    completion = concat_n_completions(
                        completion_i_s, self.mg.completion_sample_templates[v]
                    )
                else:
                    get_input_completion = PromptFn(
                        test_sample[k],
                        completion_fn=self.completion_fn,
                        max_tokens=self.max_tokens,
                    )
                    completion, _ = get_input_completion()
                completions[v] = completion
        except openai.error.InvalidRequestError:
            self.invalid_request_during_completion += 1
            return

        # run modelgraded eval
        metrics = {}
        prompt = self.mg.format(**completions, **test_sample)
        try:
            choice = self.mg.classify(
                prompt=prompt,
                completion_fn=self.eval_completion_fn,
                max_tokens=self.max_tokens,
            )
        except openai.error.InvalidRequestError:
            logging.warn(f"Invalid request during evaluation: {prompt}")
            self.invalid_request_during_evaluation += 1
            return
        metrics[CHOICE_KEY] = choice

        # run metaeval if requested
        if self.metaeval:
            assert (
                CHOICE_KEY in test_sample
            ), f"Missing label for metric '{CHOICE_KEY}' in sample {test_sample.keys()}"
            metrics[CHOICE_KEY + "_metascore"] = choice == test_sample[CHOICE_KEY]

        evals.record.record_metrics(**metrics)

        return choice

    def run(self, recorder):
        samples = self.get_samples()

        self.eval_all_samples(recorder, samples)
        record_metrics = {}
        record_metrics["invalid_request_during_completion"] = self.invalid_request_during_completion
        record_metrics["invalid_request_during_evaluation"] = self.invalid_request_during_evaluation

        all_sample_metrics = recorder.get_metrics()
        if not all_sample_metrics:
            return record_metrics

        chosen = [m[CHOICE_KEY] for m in all_sample_metrics if CHOICE_KEY in m]
        # if there is a best choice, compute the score
        if self.mg.choice_scores:
            # assumption: each INVALID_STR contributes the lowest score
            lowest_score = min(self.mg.choice_scores.values())
            scores = [
                self.mg.choice_scores[choice] if choice != INVALID_STR else lowest_score
                for choice in chosen
            ]
            record_metrics[f"score/{CHOICE_KEY}"] = sum(scores) / len(all_sample_metrics)
        # compute the counts and ratios
        counts = dict(Counter(chosen))
        missing_samples = len(all_sample_metrics) - len(chosen)
        if missing_samples:
            counts["__missing_samples__"] = missing_samples
        record_metrics.update({f"counts/{CHOICE_KEY}/{k}": v for k, v in counts.items()})
        if self.metaeval:
            metascores = [
                m[CHOICE_KEY + "_metascore"] for m in all_sample_metrics if CHOICE_KEY in m
            ]
            record_metrics[f"metascore/{CHOICE_KEY}"] = sum(metascores) / len(all_sample_metrics)

        return record_metrics
