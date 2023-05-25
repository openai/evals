"""
Generic eval that uses a prompt + classification.
"""
from collections import Counter
from random import Random
from typing import Any, Optional, Union

import evals
import evals.record
from evals.elsuite.modelgraded.classify_utils import classify, sample_and_concat_n_completions
from evals.elsuite.utils import PromptFn, scrub_formatting_from_prompt


class ModelBasedClassify(evals.Eval):
    def __init__(
        self,
        modelgraded_spec: str,
        *args,
        modelgraded_spec_args: Optional[dict[str, dict[str, str]]] = None,
        sample_kwargs: Optional[dict[str, Any]] = None,
        eval_kwargs: Optional[dict[str, Any]] = None,
        multicomp_n: Union[int, str] = 1,
        eval_type: Optional[str] = None,
        metaeval: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # treat last completion_fn as eval_completion_fn
        self.eval_completion_fn = self.completion_fns[-1]
        if len(self.completion_fns) > 1:
            self.completion_fns = self.completion_fns[:-1]
        n_models = len(self.completion_fns)
        self.sample_kwargs = {"max_tokens": 1024}
        self.sample_kwargs.update(sample_kwargs or {})
        self.eval_kwargs = {"max_tokens": 1024}
        self.eval_kwargs.update(eval_kwargs or {})
        self.metaeval = metaeval
        self.modelgraded_spec_args = modelgraded_spec_args or {}
        self.eval_type = eval_type
        if multicomp_n == "from_models":
            assert n_models > 1
            self.multicomp_n = n_models
        else:
            assert isinstance(multicomp_n, int)
            self.multicomp_n = multicomp_n
        if len(self.completion_fns) > 1:
            assert self.multicomp_n == n_models

        self.mg = self.registry.get_modelgraded_spec(modelgraded_spec)

    def eval_sample(self, test_sample: dict, rng: Random) -> None:
        """Evaluate a single sample.

        Recorded metrics are always: one of the self.choice_strings, or "__invalid__".
        """
        # process test_sample
        for k in self.mg.input_outputs:
            test_sample[k] = scrub_formatting_from_prompt(test_sample[k])

        # run policy completions
        completions = {}
        for k, v in self.mg.input_outputs.items():
            if v in test_sample:  # test_sample already has completion, skip.
                continue
            if self.multicomp_n > 1:
                completion = sample_and_concat_n_completions(
                    self.completion_fns,
                    prompt=test_sample[k],
                    template_i=self.mg.output_template,
                    sample_kwargs=self.sample_kwargs,
                    n=self.multicomp_n,
                )
            else:
                get_input_completion = PromptFn(
                    test_sample[k], completion_fn=self.completion_fn, **self.sample_kwargs
                )
                completion, _ = get_input_completion()
            completions[v] = completion

        # run modelgraded eval
        metrics = {}
        choice, info = classify(
            mg=self.mg,
            completion_fn=self.eval_completion_fn,
            completion_kwargs=self.eval_kwargs,
            eval_type=self.eval_type,
            n=self.multicomp_n,
            format_kwargs={**completions, **test_sample, **self.modelgraded_spec_args},
        )
        metrics.update(dict(choice=choice, score=info["score"]))

        # run metaeval if requested
        if self.metaeval:
            assert "choice" in test_sample
            metrics["metascore"] = choice == test_sample["choice"]

        evals.record.record_metrics(**metrics)

        return choice

    def run(self, recorder):
        samples = self.get_samples()

        self.eval_all_samples(recorder, samples)
        record_metrics = {}

        all_sample_metrics = recorder.get_metrics()
        if not all_sample_metrics:
            return record_metrics

        # record the counts
        choices = [m["choice"] for m in all_sample_metrics]
        counts = dict(Counter(choices))
        record_metrics.update({f"counts/{k}": v for k, v in counts.items()})

        # record the scores
        scores = [m["score"] for m in all_sample_metrics if m["score"] is not None]
        if scores:
            record_metrics[f"score"] = sum(scores) / len(scores)
        metascores = [m["metascore"] for m in all_sample_metrics if "metascore" in m]
        if metascores:
            record_metrics[f"metascore"] = sum(metascores) / len(metascores)

        return record_metrics
