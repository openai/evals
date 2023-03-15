from typing import Any

import numpy as np

import evals
import evals.elsuite.utils
import evals.metrics
from evals.record import RecorderBase


class ListMatch(evals.Eval):
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

        tools_predicted = set((item.removeprefix("- ") for item in sampled.splitlines()))
        tools_ground_truth = set(sample["ideal"])

        num_same = len(tools_predicted.intersection(tools_ground_truth))

        if num_same == 0:
            f1 = 0
            recall = 0
            precision = 0
        else:
            precision = 1.0 * num_same / len(tools_predicted)
            recall = 1.0 * num_same / len(tools_ground_truth)
            f1 = (2 * precision * recall) / (precision + recall)

        evals.record.record_metrics(
            accuracy=float(tools_predicted == tools_ground_truth),
            f1_score=f1,
            recall=recall,
            precision=precision,
        )

    def run(self, recorder: RecorderBase):
        samples = evals.get_jsonl(self.samples_jsonl)
        self.eval_all_samples(recorder, samples)

        return {
            "precision": np.mean(recorder.get_scores("precision")),
            "recall": np.mean(recorder.get_scores("recall")),
            "accuracy": np.mean(recorder.get_scores("accuracy")),
            "f1_score": np.mean(recorder.get_scores("f1_score")),
        }
