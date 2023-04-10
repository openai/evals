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
from evals.base import ModelSpec
from evals.elsuite.modelgraded.base import ModelGradedSpec
from evals.elsuite.modelgraded.classify_utils import (
    CHOICE_KEY,
    INVALID_STR,
    MATCH_FNS,
    concat_n_completions,
    get_choice,
)
from evals.elsuite.utils import PromptFn, scrub_formatting_from_prompt


class ModelBasedClassify(evals.Eval):
    invalid_request_during_completion = 0
    invalid_request_during_evaluation = 0

    def __init__(
        self,
        model_specs: evals.ModelSpecs,
        samples_jsonl: str,
        modelgraded_spec: str,
        *args,
        match_fn: str = "starts_or_endswith",
        max_tokens: int = 1024,
        multicomp_n: Union[int, str] = 1,
        multicomp_temperature: float = 0.4,
        samples_renamings: Optional[dict[str, str]] = None,
        eval_type: Optional[str] = None,
        eval_model: str = "gpt-3.5-turbo",
        metaeval: bool = False,
        modelgraded_spec_args: Optional[dict[str, dict[str, str]]] = None,
        **kwargs,
    ):
        super().__init__(model_specs, *args, **kwargs)
        n_models = len(self.model_specs.completions)
        self.max_tokens = max_tokens
        self.samples_jsonl = samples_jsonl
        self.match_fn = MATCH_FNS[match_fn]
        self.metaeval = metaeval
        if multicomp_n == "from_models":
            assert n_models > 1, f"multicomp_n='from_models' but only 1 model is specified."
            self.multicomp_n = n_models
        else:
            assert isinstance(
                multicomp_n, int
            ), f"multicomp_n={multicomp_n} must be an int or 'from_models'."
            self.multicomp_n = multicomp_n
        self.multicomp_temperature = multicomp_temperature
        self.samples_renamings = samples_renamings or {}

        # check if multiple models are specified
        if len(self.model_specs.completions) > 1:
            assert (
                self.multicomp_n == n_models
            ), f"multicomp_n={self.multicomp_n} must be equal to the number of models={len(self.model_specs.completions)} if multiple models are specified."

        if self.model_spec.name == "dummy-completion" or self.model_spec.name == "dummy-chat":
            self.eval_modelspec = self.model_spec
        else:
            self.eval_modelspec = ModelSpec(name=eval_model, model=eval_model, is_chat=True)

        spec_kwargs = {"multicomp_n": self.multicomp_n}
        if modelgraded_spec_args:
            spec_kwargs["args"] = modelgraded_spec_args
        self.mg: ModelGradedSpec = self.registry.get_modelgraded_spec(
            modelgraded_spec, **spec_kwargs
        )
        if eval_type:
            self.mg.append_answer_prompt(eval_type)

    def eval_sample(self, test_sample: dict, rng: Random) -> None:
        """Evaluate a single sample.

        Recorded metrics are always: one of the self.choice_strings, or "__invalid__".
        """
        if self.samples_renamings:
            test_sample = {self.samples_renamings.get(k, k): v for k, v in test_sample.items()}
        if self.multicomp_n > 1:
            test_sample["n"] = self.multicomp_n
        completions = {}
        if self.metaeval:
            # assert outputs exist in the data
            for v in self.mg.input_outputs.values():
                assert v in test_sample, f"Missing output '{v}' in sample {test_sample.keys()}"
                completions[v] = test_sample[v]
        # remove outputs from the data
        test_sample = {
            k: v for k, v in test_sample.items() if k not in list(self.mg.input_outputs.values())
        }
        for k in self.mg.input_outputs:
            test_sample[k] = scrub_formatting_from_prompt(test_sample[k])

        if not self.metaeval:
            try:
                for k, v in self.mg.input_outputs.items():
                    if self.multicomp_n > 1 and v in self.mg.completion_sample_templates:
                        completion_i_s = []
                        for i in range(self.multicomp_n):
                            if len(self.model_specs.completions) > 1:
                                # use a separate model for each completion
                                model_spec = self.model_specs.completions[i]
                            else:
                                # use the single model for all completions
                                model_spec = self.model_spec
                            get_input_completion = PromptFn(
                                test_sample[k],
                                model_spec=model_spec,
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
                            model_spec=self.model_spec,
                            max_tokens=self.max_tokens,
                        )
                        completion, _ = get_input_completion()
                    completions[v] = completion
            except openai.error.InvalidRequestError:
                self.invalid_request_during_completion += 1
                return

        metrics = {}
        if self.mg.expanded_args_dict and len(self.mg.expanded_args_dict) > 1:
            args_dict = self.mg.expanded_args_dict
        elif self.mg.expanded_args_dict and len(self.mg.expanded_args_dict) == 1:
            # if there is only one combination, don't bother with the metric name
            args_dict = {CHOICE_KEY: v for v in self.mg.expanded_args_dict.values()}
        else:
            args_dict = {CHOICE_KEY: {}}
        for metric, args in args_dict.items():
            args = {k: v[1] for k, v in args.items()}
            prompt = self.mg.format(**args, **completions, **test_sample)
            evaluate = PromptFn(
                prompt,
                model_spec=self.eval_modelspec,
                max_tokens=self.max_tokens,
            )
            try:
                evaluation, _ = evaluate(skip_format=True)
            except openai.error.InvalidRequestError:
                logging.warn(f"Invalid request during evaluation: {prompt}")
                self.invalid_request_during_evaluation += 1
                return
            choice = get_choice(
                evaluation, self.mg.eval_type, self.match_fn, self.mg.choice_strings
            )
            if choice == INVALID_STR:
                logging.warn(
                    f"Choices {self.mg.choice_strings} not parsable for {self.mg.eval_type}: {evaluation}"
                )
            metrics[metric] = choice
            if self.metaeval:
                assert (
                    metric in test_sample
                ), f"Missing label for metric '{metric}' in sample {test_sample.keys()}"
                metrics[metric + "_metascore"] = choice == test_sample[metric]

        evals.record.record_metrics(**metrics)

        return choice

    def run(self, recorder):
        samples = evals.get_jsonl(self.samples_jsonl)

        self.eval_all_samples(recorder, samples)
        record_metrics = {}
        record_metrics["invalid_request_during_completion"] = self.invalid_request_during_completion
        record_metrics["invalid_request_during_evaluation"] = self.invalid_request_during_evaluation

        all_sample_metrics = recorder.get_metrics()
        if not all_sample_metrics:
            return record_metrics

        if self.mg.expanded_args_dict and len(self.mg.expanded_args_dict) > 1:
            metrics = sorted(self.mg.expanded_args_dict)
        else:
            metrics = [CHOICE_KEY]
        for metric in metrics:
            chosen = [m[metric] for m in all_sample_metrics if metric in m]
            # if there is a best choice, compute the score
            if self.mg.choice_scores:
                # assumption: each INVALID_STR contributes the lowest score
                lowest_score = min(self.mg.choice_scores.values())
                scores = [
                    self.mg.choice_scores[choice] if choice != INVALID_STR else lowest_score
                    for choice in chosen
                ]
                record_metrics[f"score/{metric}"] = sum(scores) / len(all_sample_metrics)
            # compute the counts and ratios
            counts = dict(Counter(chosen))
            missing_samples = len(all_sample_metrics) - len(chosen)
            if missing_samples:
                counts["__missing_samples__"] = missing_samples
            record_metrics.update({f"counts/{metric}/{k}": v for k, v in counts.items()})
            if self.metaeval:
                metascores = [m[metric + "_metascore"] for m in all_sample_metrics if metric in m]
                record_metrics[f"metascore/{metric}"] = sum(metascores) / len(all_sample_metrics)

        return record_metrics
