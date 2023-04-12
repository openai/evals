import numpy as np

import evals
from evals.api import CompletionFn
from evals.elsuite import utils
from evals.record import RecorderBase

from graphql import parse
from graphql import ast_to_dict
from graphql import strip_ignored_characters

from deepdiff import DeepDiff

class GraphQL(evals.Eval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        samples_jsonl: str,
        fuzzy = False,
        *args,
        max_tokens: int = 500,
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert len(completion_fns) == 1, "GraphQLMatch only supports one completion fn"
        self.max_tokens = max_tokens
        self.samples_jsonl = samples_jsonl
        self.fuzzy = fuzzy


    def eval_sample(self, test_sample, rng):
        del rng
        prompt, correct_answers = test_sample["input"], test_sample["ideal"]
        result = self.completion_fn(
            prompt=prompt,
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


    def match_graphql( self, answer: str, truth: str ) -> bool:
        try:
            truth_ast = parse(strip_ignored_characters(truth))
            truth_dict = ast_to_dict(truth_ast)
            answer_ast = parse(strip_ignored_characters(answer))
            answer_dict = ast_to_dict(answer_ast)
            diff = {}
            if self.fuzzy:
                diff = DeepDiff(truth_dict, answer_dict, ignore_order=True, exclude_paths=["root['definitions'][0]['name']"], exclude_regex_paths=r".\['alias'\].")
            else:
                diff = DeepDiff(truth_dict, answer_dict)

            return not bool(diff)
        except Exception as e:
            print("Exception ---- ", e)
            return False

    def run(self, recorder: RecorderBase):
        samples = self.get_samples()
        self.eval_all_samples(recorder, samples)

        return {
            "accuracy": np.mean(recorder.get_scores("accuracy")),
            "f1_score": np.mean(recorder.get_scores("f1_score")),
        }
