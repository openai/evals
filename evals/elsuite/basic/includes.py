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
        *args,
        max_tokens: int = 500,
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert len(completion_fns) == 1, "includes only supports one completion fn"
        self.max_tokens = max_tokens
        self.samples_jsonl = samples_jsonl

    def eval_sample(self, sample: Any, *_):
        prompt = sample["input"]
        result = self.completion_fn(
            prompt=prompt,
            max_tokens=self.max_tokens,
        )
        sampled = result.get_completions()[0]
        evals.record.record_sampling(prompt=prompt, sampled=sampled)

        includes_answer = any([utils.get_answer(sampled, ref) for ref in sample["ideal"]])
        evals.record.record_metrics(accuracy=float(includes_answer))
        return includes_answer

    def run(self, recorder):
        samples = self.get_samples()
        self.eval_all_samples(recorder, samples)
        events = recorder.get_scores("accuracy")
        return {
            "accuracy": np.mean(events),
        }
