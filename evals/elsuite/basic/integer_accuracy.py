import numpy as np

import evals
from evals.elsuite import utils
from evals.record import RecorderBase


class IntegerAccuracy(evals.Eval):
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

    def eval_sample(self, test_sample, rng):
        prompt, correct_answer = test_sample["input"], test_sample["ideal"]
        generated_answer = evals.sample_freeform(
            self.model_spec,
            prompt,
            temperature=0.0,
            max_tokens=16,
        )

        accuracy_score = utils.integer_accuracy(generated_answer, correct_answer)

        evals.record.record_metrics(accuracy=accuracy_score)

        return accuracy_score

    def run(self, recorder: RecorderBase):
        samples = evals.get_jsonl(self.samples_jsonl)
        self.eval_all_samples(recorder, samples)

        return {
            "accuracy": np.mean(recorder.get_scores("accuracy")),
        }
