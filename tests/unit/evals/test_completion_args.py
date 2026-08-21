import pytest

from evals.cli.oaieval import parse_completion_args
from evals.completion_fns.openai import OpenAIChatCompletionFn, OpenAICompletionFn


def test_parse_completion_args_preserves_json_types() -> None:
    parsed = parse_completion_args(
        'temperature=0.5,max_tokens=10,stream=false,stop=["END","STOP"],metadata={"a":1,"b":2}'
    )

    assert parsed == {
        "temperature": 0.5,
        "max_tokens": 10,
        "stream": False,
        "stop": ["END", "STOP"],
        "metadata": {"a": 1, "b": 2},
    }


def test_parse_completion_args_splits_only_first_equals_sign() -> None:
    assert parse_completion_args("base_url=https://example.test?a=b") == {
        "base_url": "https://example.test?a=b"
    }


def test_parse_completion_args_preserves_apostrophes_in_literal_strings() -> None:
    assert parse_completion_args("label=don't-stop") == {"label": "don't-stop"}


def test_parse_completion_args_rejects_invalid_syntax() -> None:
    with pytest.raises(ValueError, match="key=value"):
        parse_completion_args("temperature")


def test_chat_completion_constructor_forwards_api_options() -> None:
    completion_fn = OpenAIChatCompletionFn(
        model="gpt-4",
        temperature=0.5,
        max_tokens=64,
        registry=object(),
    )

    assert completion_fn.extra_options == {"temperature": 0.5, "max_tokens": 64}


def test_legacy_completion_constructor_forwards_api_options() -> None:
    completion_fn = OpenAICompletionFn(
        model="text-davinci-003",
        temperature=0.5,
        max_tokens=64,
        registry=object(),
    )

    assert completion_fn.extra_options == {"temperature": 0.5, "max_tokens": 64}


def test_explicit_extra_options_override_convenience_kwargs_without_mutation() -> None:
    extra_options = {"temperature": 0.2, "stop": ["END"]}

    completion_fn = OpenAIChatCompletionFn(
        model="gpt-4",
        extra_options=extra_options,
        temperature=0.8,
        max_tokens=32,
    )

    assert completion_fn.extra_options == {
        "temperature": 0.2,
        "stop": ["END"],
        "max_tokens": 32,
    }
    assert extra_options == {"temperature": 0.2, "stop": ["END"]}
