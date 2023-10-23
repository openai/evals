import json
import random
from typing import Any, Dict, List, Mapping, Union, cast

import numpy as np

import evals
from evals.api import CompletionFn
from evals.record import RecorderBase


def json_match(sampled_json: Any, correct_json: Any) -> bool:
    """Return True if the sampled completion in JSON format
    matches a correct answer, component by component"""
    if sampled_json is None or correct_json is None:
        # Missing values are never correct
        return False
    if isinstance(sampled_json, dict):
        if isinstance(correct_json, dict):
            sample = cast(Mapping[str, Any], sampled_json)
            correct = cast(Mapping[str, Any], correct_json)
            all_keys = set(sample.keys()) | set(correct.keys())
            return all(json_match(sample.get(key), correct.get(key)) for key in all_keys)
        else:
            return False
    elif isinstance(sampled_json, list):
        if isinstance(correct_json, list):
            slist = cast(List[Any], sampled_json)
            clist = cast(List[Any], correct_json)
            if len(slist) != len(clist):
                # Lists must have the same length
                return False
            return all(json_match(s, c) for s, c in zip(slist, clist))
        else:
            return False
    # Not a structured item: do a direct comparison
    return sampled_json == correct_json


class JsonMatch(evals.Eval):

    """Compares a JSON completion with one or more ideal answers,
    also coded in JSON. The decoded JSON objects are compared
    elementwise and must match exactly."""

    def __init__(
        self,
        completion_fns: list[CompletionFn],
        samples_jsonl: str,
        *args: Any,
        max_tokens: int = 512,  # Increase this for longer JSON completions
        **kwargs: Any,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert len(completion_fns) == 1, "JsonMatch only supports one completion fn"
        self.max_tokens = max_tokens
        self.samples_jsonl = samples_jsonl

    def eval_sample(self, sample: Any, rng: random.Random):
        del rng

        assert isinstance(sample, dict), "sample must be a dict"
        assert "input" in sample, "sample must have an 'input' key"
        assert "ideal" in sample, "sample must have an 'ideal' key"

        prompt = cast(str, sample["input"])
        correct_answers = cast(Union[str, List[str]], sample["ideal"])
        if not isinstance(correct_answers, list):
            correct_answers = [correct_answers]

        result = self.completion_fn(
            prompt=prompt,
            temperature=0.0,  # Q: why are these hardcoded?
            max_tokens=self.max_tokens,
        )
        sampled = result.get_completions()[0]

        sampled_json: Any
        try:
            sampled_json = json.loads(sampled)
        except ValueError:
            # If the sampled string is not valid JSON, it will never match
            sampled_json = None

        # Allow the following to raise ValueError; the correct answers
        # should always be valid JSON
        correct_json = [json.loads(correct_answer) for correct_answer in correct_answers]

        matches = [json_match(sampled_json, cj) for cj in correct_json]

        evals.record.record_match(
            True in matches,
            expected=correct_answers,
            picked=[sampled for i in range(len(correct_answers)) if matches[i]],
        )
        evals.record.record_metrics(
            accuracy=float(True in matches),
        )

    def run(self, recorder: RecorderBase) -> Dict[str, float]:
        samples = self.get_samples()
        self.eval_all_samples(recorder, samples)

        return {
            "accuracy": np.mean(recorder.get_scores("accuracy")),
        }
