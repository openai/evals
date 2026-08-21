from unittest.mock import patch

import pytest

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
