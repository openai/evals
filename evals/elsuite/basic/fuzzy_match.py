import numpy as np

import evals
from evals.api import CompletionFn
from evals.elsuite import utils
from evals.record import RecorderBase


class FuzzyMatch(evals.Eval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        samples_jsonl: str,
        *args,
        max_tokens: int = 16,  # was 500 before, but not used, whereas 16 was hardcoded in eval_sample(...)
        preserve_punct: str = None,  # whitelist punctionation to be preserved through utils.normalize(...)
        multiline: bool = None,  # Q: utils.normalize(...) will only compare the first line of content by default, is that well known?
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert len(completion_fns) == 1, "FuzzyMatch only supports one completion fn"
        self.max_tokens = max_tokens
        self.samples_jsonl = samples_jsonl
        self.preserve_punct = preserve_punct
        self.multiline = multiline

    def eval_sample(self, sample, rng):
        del rng
        prompt, correct_answers = sample["input"], sample["ideal"]
        if not isinstance(correct_answers, list):
            correct_answers = [correct_answers]

        result = self.completion_fn(
            prompt=prompt,
            temperature=0.0,  # Q: why are these hardcoded?
            max_tokens=self.max_tokens,
        )
        sampled = result.get_completions()[0]

        normalize_kwargs = {"preserve_punct": self.preserve_punct, "multiline": self.multiline}
        matches = [
            utils.fuzzy_match(sampled, correct_answer, **normalize_kwargs)
            for correct_answer in correct_answers
        ]

        evals.record.record_match(
            True in matches,
            expected=correct_answers,
            picked=[sampled for i in range(len(correct_answers)) if matches[i]],
        )
        evals.record.record_metrics(
            accuracy=float(True in matches),
            f1_score=utils.f1_score(sampled, correct_answers, **normalize_kwargs),
        )

    def run(self, recorder: RecorderBase):
        samples = self.get_samples()
        self.eval_all_samples(recorder, samples)

        return {
            "accuracy": np.mean(recorder.get_scores("accuracy")),
            "f1_score": np.mean(recorder.get_scores("f1_score")),
        }
