from typing import Any

import numpy as np

import evals
import evals.elsuite.utils
import evals.metrics


class Includes(evals.Eval):
    def __init__(
        self,
        model_specs: evals.ModelSpecs,
        samples_jsonl: str,
        *args,
        max_tokens: int = 500,
        **kwargs,
    ):
        super().__init__(model_specs, *args, **kwargs)
        self.max_tokens = max_tokens
        self.samples_jsonl = samples_jsonl

    def eval_sample(self, sample: Any, *_):
        sampled = evals.sample_freeform(
            self.model_spec, sample["input"], max_tokens=self.max_tokens
        )
        includes_answer = any(
            [evals.elsuite.utils.get_answer(sampled, ref) for ref in sample["ideal"]]
        )
        evals.record.record_metrics(accuracy=float(includes_answer))
        return includes_answer

    def run(self, recorder):
        samples = evals.get_jsonl(self.samples_jsonl)
        self.eval_all_samples(recorder, samples)
        events = recorder.get_scores("accuracy")
        return {
            "accuracy": np.mean(events),
        }
