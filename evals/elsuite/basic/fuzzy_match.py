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
        completion_fn: evals.CompletionFn = evals.OpenAIChatCompletionFn(),
        **kwargs,
    ):
        super().__init__(model_specs, *args, **kwargs)
        self.max_tokens = max_tokens
        self.samples_jsonl = samples_jsonl
        self._completion_fn = completion_fn

    def eval_sample(self, test_sample, rng):
        del rng
        prompt, correct_answers = test_sample["input"], test_sample["ideal"]
        result = self._completion_fn(
            prompt=prompt,
            temperature=0.0,  # Q: why are these hardcoded?
            max_tokens=16,
            model_spec=self.model_spec,
        )
        sampled = result.get_completions()[0]
        evals.record.record_sampling(prompt=result.prompt, sampled=sampled)

        matches = [utils.fuzzy_match(sampled, correct_answer) for correct_answer in correct_answers]
        evals.record.record_match(
            True in matches,
            expected=correct_answers,
            picked=[sampled for i in range(len(correct_answers)) if matches[i]],
        )
        evals.record.record_metrics(
            accuracy=float(True in matches),
            f1_score=utils.f1_score(sampled, correct_answers),
        )

    def run(self, recorder: RecorderBase):
        samples = self.get_samples()
        self.eval_all_samples(recorder, samples)

        return {
            "accuracy": np.mean(recorder.get_scores("accuracy")),
            "f1_score": np.mean(recorder.get_scores("f1_score")),
        }
