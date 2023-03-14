import numpy as np

import evals
from evals.elsuite import utils
from evals.record import RecorderBase


class FuzzyMatch(evals.Eval):
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
        prompt, correct_answers = test_sample["input"], test_sample["ideal"]
        generated_answer = evals.sample_freeform(
            self.model_spec,
            prompt,
            temperature=0.0,
            max_tokens=16,
        )
        matches = [
            utils.fuzzy_match(generated_answer, correct_answer)
            for correct_answer in correct_answers
        ]
        evals.record.record_match(
            True in matches,
            expected=correct_answers,
            picked=[generated_answer for i in range(len(correct_answers)) if matches[i]],
        )
        evals.record.record_metrics(
            accuracy=float(True in matches),
            f1_score=utils.f1_score(generated_answer, correct_answers),
        )

    def run(self, recorder: RecorderBase):
        samples = evals.get_jsonl(self.samples_jsonl)
        self.eval_all_samples(recorder, samples)

        return {
            "accuracy": np.mean(recorder.get_scores("accuracy")),
            "f1_score": np.mean(recorder.get_scores("f1_score")),
        }
