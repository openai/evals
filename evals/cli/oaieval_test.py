import os

os.environ.setdefault("OPENAI_API_KEY", "dummy")

from evals.cli.oaieval import parse_key_value_args
from evals.registry import Registry


def test_parse_key_value_args_coerces_numbers() -> None:
    parsed = parse_key_value_args("temperature=0.5,max_tokens=10,label=test")

    assert parsed == {"temperature": 0.5, "max_tokens": 10, "label": "test"}


def test_registry_make_completion_fn_passes_completion_args_to_chat_models() -> None:
    completion_fn = Registry().make_completion_fn("gpt-3.5-turbo", temperature=0.5)

    assert completion_fn.extra_options == {"temperature": 0.5}