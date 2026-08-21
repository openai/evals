from typing import Any, Union

from evals.prompt.base import is_chat_prompt
from evals.record import record_match

from .match import Match


def record_and_check_exact_match(
    sampled: str,
    expected: Union[str, list[str], tuple[str, ...]],
):
    """Record and return the expected option that exactly equals ``sampled``."""
    if isinstance(expected, tuple):
        expected = list(expected)
    elif not isinstance(expected, list):
        expected = [expected]

    picked = next((option for option in expected if sampled == option), None)
    record_match(
        picked is not None,
        expected=expected,
        picked=picked,
        sampled=sampled,
        options=expected,
    )
    return picked


class ExactMatch(Match):
    """Match only when the complete model response equals a reference answer."""

    def eval_sample(self, sample: Any, *_):
        assert isinstance(sample, dict), "sample must be a dict"
        assert "input" in sample, "sample must have an 'input' key"
        assert "ideal" in sample, "sample must have an 'ideal' key"
        assert isinstance(sample["ideal"], str) or isinstance(
            sample["ideal"], list
        ), "sample['ideal'] must be a string or list of strings"

        prompt = sample["input"]
        if self.num_few_shot > 0:
            assert is_chat_prompt(sample["input"]), "few shot requires chat prompt"
            prompt = sample["input"][:-1]
            for s in self.few_shot[: self.num_few_shot]:
                prompt += s["sample"]
            prompt += sample["input"][-1:]

        result = self.completion_fn(
            prompt=prompt,
            temperature=0.0,
        )
        sampled = result.get_completions()[0]

        return record_and_check_exact_match(sampled=sampled, expected=sample["ideal"])
