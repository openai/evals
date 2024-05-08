import numpy as np
from pyparsing import Any

import evals
import evals.elsuite.text_compression.compression as compression
import evals.metrics
from evals.api import CompletionFn
from evals.eval import Eval
from evals.record import RecorderBase


class TextCompression(Eval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        samples_jsonl: str,
        *args,
        prompt_version: str = "3.0",
        reconstruction_error_metric: str = "character_error_rate_cap1",
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert len(completion_fns) == 1, "TextCompression only supports one completion fn"
        self.samples_jsonl = samples_jsonl
        self.prompt_version = prompt_version
        self.reconstruction_error_metric = reconstruction_error_metric

    def eval_sample(self, sample: Any, *_):
        assert isinstance(sample, dict), "sample must be a dict"
        assert "text" in sample, "sample must have a text key"
        assert isinstance(sample["text"], str), "sample text must be a string"

        payload = sample["text"]
        results = compression.run_eval(
            payload=payload,
            completion_fn=self.completion_fns[0],
            prompt_version=self.prompt_version,
            metadata=sample,
        )

        evals.record.record_metrics(
            compression_ratio=results["compression_ratio"],
            compression_ratio_cap1=results["compression_ratio_cap1"],
            reconstruction_error=results[self.reconstruction_error_metric],
            character_error_rate_cap1=results["character_error_rate_cap1"],
            semantic_distance=results["semantic_distance"],
        )

    def _run_impl(self, recorder: RecorderBase):
        samples = self.get_samples()
        self.eval_all_samples(recorder, samples)
        metrics = recorder.get_metrics()
        return {
            "compression_ratio": np.mean([d["compression_ratio"] for d in metrics]),
            "compression_ratio_cap1": np.mean([d["compression_ratio_cap1"] for d in metrics]),
            "reconstruction_error": np.mean([d["reconstruction_error"] for d in metrics]),
            "character_error_rate_cap1": np.mean([d["character_error_rate_cap1"] for d in metrics]),
            "semantic_distance": np.mean([d["semantic_distance"] for d in metrics]),
        }
