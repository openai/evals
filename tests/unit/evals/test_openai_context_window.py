from unittest.mock import patch

import pytest
import tiktoken

from evals.completion_fns import openai as completion_module
from evals.completion_fns.openai import OpenAIChatCompletionFn, OpenAICompletionFn


def test_chat_completion_rejects_oversized_prompt_before_api_request() -> None:
    completion_fn = OpenAIChatCompletionFn(
        model="gpt-3.5-turbo",
        api_key="test",
        n_ctx=5,
    )
    prompt = [{"role": "user", "content": "This prompt is definitely longer than five tokens."}]

    with patch(
        "evals.completion_fns.openai.openai_chat_completion_create_retrying"
    ) as create_completion:
        with pytest.raises(ValueError, match="configured context window of 5 tokens"):
            completion_fn(prompt)

    create_completion.assert_not_called()


def test_text_completion_rejects_oversized_prompt_before_api_request() -> None:
    completion_fn = OpenAICompletionFn(
        model="text-davinci-003",
        api_key="test",
        n_ctx=1,
    )

    with patch(
        "evals.completion_fns.openai.openai_completion_create_retrying"
    ) as create_completion:
        with pytest.raises(ValueError, match="configured context window of 1 tokens"):
            completion_fn("two words")

    create_completion.assert_not_called()


def test_unknown_context_window_skips_preflight() -> None:
    completion_fn = OpenAICompletionFn(
        model="custom-model",
        api_key="test",
        n_ctx=None,
    )

    # The request helper is expected to be reached when no context-window metadata
    # is available. Raising here avoids needing a real API response in this unit test.
    with patch(
        "evals.completion_fns.openai.openai_completion_create_retrying",
        side_effect=RuntimeError("request reached"),
    ) as create_completion:
        with pytest.raises(RuntimeError, match="request reached"):
            completion_fn("a long prompt that cannot be preflighted")

    create_completion.assert_called_once()


@pytest.fixture(autouse=True)
def synthetic_encoding(monkeypatch):
    # Use the real tokenizer with a synthetic byte vocabulary, without downloads.
    encoding = tiktoken.Encoding(
        name="context-window-test",
        pat_str=r"(?s).",
        mergeable_ranks={bytes([value]): value for value in range(256)},
        special_tokens={"<|endoftext|>": 256},
    )
    monkeypatch.setattr(completion_module.tiktoken, "encoding_for_model", lambda _: encoding)
    monkeypatch.setattr(completion_module.tiktoken, "get_encoding", lambda _: encoding)
    return encoding


def test_independent_text_prompts_are_not_summed():
    fn = OpenAICompletionFn(model="text-davinci-003", api_key="test", n_ctx=4)
    with patch.object(
        completion_module,
        "openai_completion_create_retrying",
        side_effect=RuntimeError("request reached"),
    ):
        with pytest.raises(RuntimeError, match="request reached"):
            fn(["four", "five"])


def test_one_oversized_batch_member_is_rejected():
    with pytest.raises(ValueError, match="context window"):
        completion_module._validate_prompt_context(["ok", "oversized"], "test", 4)


def test_special_token_text_is_counted_as_literal_input():
    assert completion_module._estimate_prompt_tokens("<|endoftext|>", "test") == len(
        "<|endoftext|>"
    )


def test_token_id_counts_do_not_load_a_tokenizer(monkeypatch):
    def fail(_):
        raise AssertionError("token IDs need no tokenizer")

    monkeypatch.setattr(completion_module, "_encoding_for_model", fail)
    assert completion_module._estimate_prompt_tokens([1, 2, 3], "unknown") == 3
    assert completion_module._estimate_prompt_tokens([[1, 2], [3]], "unknown") == 2


def test_unknown_tokenizer_is_not_replaced_with_an_unrelated_encoding(monkeypatch):
    def missing(_):
        raise KeyError("unknown model")

    monkeypatch.setattr(completion_module.tiktoken, "encoding_for_model", missing)
    completion_module._validate_prompt_context("long input", "custom-model", 1)


def test_structured_chat_content_does_not_claim_a_text_only_estimate():
    assert (
        completion_module._estimate_prompt_tokens(
            [{"role": "user", "content": [{"type": "text", "text": "hello"}]}], "test"
        )
        is None
    )
