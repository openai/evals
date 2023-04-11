import numpy as np

import evals
from evals.api import CompletionFn
from evals.elsuite import utils
from evals.record import RecorderBase


class GraphQLStrict(evals.Eval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        samples_jsonl: str,
        *args,
        max_tokens: int = 500,
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert len(completion_fns) == 1, "GraphQLMatch only supports one completion fn"
        self.max_tokens = max_tokens
        self.samples_jsonl = samples_jsonl

    def eval_sample(self, test_sample, rng):
        del rng
        prompt, correct_answers = test_sample["input"], test_sample["ideal"]
        result = self.completion_fn(
            prompt=prompt,
            temperature=0.0,  # Q: why are these hardcoded?
            max_tokens=16,
        )
        sampled = result.get_completions()[0]

        matches = [self.match_graphql(sampled, correct_answer) for correct_answer in correct_answers]
        evals.record.record_match(
            True in matches,
            expected=correct_answers,
            picked=[sampled for i in range(len(correct_answers)) if matches[i]],
        )
        evals.record.record_metrics(
            accuracy=float(True in matches),
            f1_score=utils.f1_score(sampled, correct_answers),
        )


    def match_graphql( self, q1: str, q2: str ) -> bool:
        # Todo: add a lib that takes queries as input and compares them to return output.
        print("query 1", q1, "query 2", q2)
        return True

    def run(self, recorder: RecorderBase):
        samples = self.get_samples()
        self.eval_all_samples(recorder, samples)

        return {
            "accuracy": np.mean(recorder.get_scores("accuracy")),
            "f1_score": np.mean(recorder.get_scores("f1_score")),
        }
