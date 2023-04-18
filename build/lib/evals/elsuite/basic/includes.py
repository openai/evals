from typing import Any

import numpy as np

import evals
import evals.metrics
from evals.api import CompletionFn
from evals.elsuite import utils


class Includes(evals.Eval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        samples_jsonl: str,
        ignore_case: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert len(completion_fns) == 1, "Includes only supports one completion fn"
        self.samples_jsonl = samples_jsonl
        self.ignore_case = ignore_case

    def eval_sample(self, sample: Any, *_):
        prompt = sample["input"]
        result = self.completion_fn(
            prompt=prompt,
        )
        sampled = result.get_completions()[0]

        includes_answer = any(
            [utils.get_answer(sampled, ref, self.ignore_case) for ref in sample["ideal"]]
        )
        evals.record.record_metrics(accuracy=float(includes_answer))
        return includes_answer

    def run(self, recorder):
        samples = self.get_samples()
        self.eval_all_samples(recorder, samples)
        events = recorder.get_scores("accuracy")
        return {
            "accuracy": np.mean(events),
        }
