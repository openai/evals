import numpy as np

import re

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
        extract_gql = True,
        *args,
        max_tokens: int = 500,
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert len(completion_fns) == 1, "GraphQLMatch only supports one completion fn"
        self.max_tokens = max_tokens
        self.samples_jsonl = samples_jsonl
        self.fuzzy = fuzzy
        self.extract_gql = extract_gql


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

        gql_answer = answer
        gql_truth = truth
        if self.extract_gql:
            gql_answer = self.extract_graphQL(gql_answer)
            gql_truth = self.extract_graphQL(gql_truth)

        if not (gql_answer and gql_truth):
            return True

        if not (gql_answer or gql_truth):
            return False

        try:
            truth_ast = parse(strip_ignored_characters(gql_truth))
            truth_dict = ast_to_dict(truth_ast)
            answer_ast = parse(strip_ignored_characters(gql_answer))
            answer_dict = ast_to_dict(answer_ast)
            diff = {}
            if self.fuzzy:
                #  exclude path: ["root['definitions'][0]['name']"] for query name.
                #  exclude_regex_paths .*\['alias'\].* -> if query has any aliases, .*\['arguments'\]\[[0-9]+\]([\['value'\]\['fields'\]\[[0-9]+\])\['value'\]\['value'\] = if arguement has any filter values
                diff = DeepDiff(truth_dict, answer_dict, ignore_order=True, exclude_paths=["root['definitions'][0]['name']"], exclude_regex_paths=r".*\['alias'\].*|.*\['arguments'\]\[[0-9]+\]([\['value'\]\['fields'\]\[[0-9]+\])\['value'\]\['value'\]")
                allowed_changes = ["dictionary_item_added", "iterable_item_added"]
                for change in allowed_changes:
                    if change in diff.keys():
                        del diff[change]
            else:
                diff = DeepDiff(truth_dict, answer_dict)

            return not bool(diff)
        except Exception as e:
            print("Exception ---- ", e)
            return False

    def extract_graphQL(self, string: str) -> str:
        match_group = re.search(r'(query|mutation|subscription)\s([a-zA-Z][a-zA-Z0-9]*\s){0,1}\{', string)
        if match_group:
            gql = match_group.group()
            start_index = match_group.end() - 1
            open_braces = 1
            end_index = start_index + 1
            while open_braces > 0 and end_index < len(string):
                if string[end_index] == '{':
                    open_braces += 1
                elif string[end_index] == '}':
                    open_braces -= 1
                end_index += 1
            if open_braces == 0:
                gql += string[start_index+1:end_index].strip()
                return gql
        return None

    def run(self, recorder: RecorderBase):
        samples = self.get_samples()
        self.eval_all_samples(recorder, samples)

        return {
            "accuracy": np.mean(recorder.get_scores("accuracy")),
            "f1_score": np.mean(recorder.get_scores("f1_score")),
        }
