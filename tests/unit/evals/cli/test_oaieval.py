from typing import Any, Optional
from unittest.mock import MagicMock

from openai.types import CompletionUsage
from openai.types.completion_usage import CompletionTokensDetails, PromptTokensDetails

from evals.cli.oaieval import add_token_usage_to_result, flatten_usage


def make_usage(reasoning_tokens: Optional[int] = 32) -> CompletionUsage:
    return CompletionUsage(
        prompt_tokens=12,
        completion_tokens=48,
        total_tokens=60,
        completion_tokens_details=CompletionTokensDetails(reasoning_tokens=reasoning_tokens),
        prompt_tokens_details=PromptTokensDetails(cached_tokens=4),
    )


def make_recorder(usages: list[Any]) -> MagicMock:
    recorder = MagicMock()
    recorder.get_events.return_value = [MagicMock(data={"usage": usage}) for usage in usages]
    return recorder


def test_flatten_usage_expands_nested_details() -> None:
    flat = flatten_usage(make_usage())

    assert flat["prompt_tokens"] == 12
    assert flat["completion_tokens"] == 48
    assert flat["total_tokens"] == 60
    assert flat["completion_tokens_details.reasoning_tokens"] == 32
    assert flat["prompt_tokens_details.cached_tokens"] == 4
    # No nested model objects should survive flattening.
    assert all(not hasattr(value, "model_dump") for value in flat.values())


def test_flatten_usage_accepts_plain_dicts() -> None:
    assert flatten_usage({"prompt_tokens": 1, "details": {"cached_tokens": 2}}) == {
        "prompt_tokens": 1,
        "details.cached_tokens": 2,
    }


def test_add_token_usage_sums_nested_details() -> None:
    """Regression test for #1582: nested usage objects used to raise TypeError."""
    result: dict[str, Any] = {}
    add_token_usage_to_result(result, make_recorder([make_usage(), make_usage()]))

    assert result["usage_prompt_tokens"] == 24
    assert result["usage_completion_tokens"] == 96
    assert result["usage_total_tokens"] == 120
    # Reasoning tokens are now reported; previously they could not be aggregated at all.
    assert result["usage_completion_tokens_details.reasoning_tokens"] == 64
    assert result["usage_prompt_tokens_details.cached_tokens"] == 8


def test_add_token_usage_handles_none_values() -> None:
    """Models that omit a detail field report None, which must not break summing."""
    result: dict[str, Any] = {}
    add_token_usage_to_result(result, make_recorder([make_usage(None), make_usage(8)]))

    assert result["usage_completion_tokens_details.reasoning_tokens"] == 8


def test_add_token_usage_unions_keys_across_events() -> None:
    """A field missing from the first event must still be aggregated."""
    result: dict[str, Any] = {}
    add_token_usage_to_result(
        result,
        make_recorder([{"prompt_tokens": 1}, {"prompt_tokens": 1, "reasoning_tokens": 5}]),
    )

    assert result["usage_prompt_tokens"] == 2
    assert result["usage_reasoning_tokens"] == 5


def test_add_token_usage_skips_non_numeric_values() -> None:
    """Non-numeric usage values are dropped rather than crashing the run."""
    result: dict[str, Any] = {}
    add_token_usage_to_result(result, make_recorder([{"prompt_tokens": 3, "model": "gpt-4"}]))

    assert result["usage_prompt_tokens"] == 3
    assert "usage_model" not in result


def test_add_token_usage_preserves_existing_keys() -> None:
    result: dict[str, Any] = {"usage_prompt_tokens": 999}
    add_token_usage_to_result(result, make_recorder([{"prompt_tokens": 3}]))

    assert result["usage_prompt_tokens"] == 999
