from typing import Any

import numpy as np

import evals
import evals.metrics
from evals.elsuite import utils


class Includes(evals.Eval):
    def __init__(
        self,
        model_specs: evals.ModelSpecs,
        samples_jsonl: str,
        *args,
        max_tokens: int = 500,
        completion_fn: evals.CompletionFn = evals.OpenAICompletionFn(),
        **kwargs,
    ):
        super().__init__(model_specs, *args, **kwargs)
        self.max_tokens = max_tokens
        self.samples_jsonl = samples_jsonl
        self._completion_fn = completion_fn

    def eval_sample(self, sample: Any, *_):
        result = self._completion_fn(
            prompt=sample["input"],
            max_tokens=self.max_tokens,
            model_spec=self.model_spec,
        )
        sampled: str = evals.postprocess_sample_freeform(
            result.get_completions(), result.prompt, self.model_spec
        )

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
