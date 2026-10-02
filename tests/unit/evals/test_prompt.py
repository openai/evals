import pytest

from evals.prompt.base import (
    ChatCompletionPrompt,
    CompletionPrompt,
    chat_prompt_to_text_prompt,
    is_chat_prompt,
)


def test_empty_list_is_not_a_chat_prompt() -> None:
    assert is_chat_prompt([]) is False


def test_non_empty_list_of_messages_is_a_chat_prompt() -> None:
    assert is_chat_prompt([{"role": "user", "content": "hello"}]) is True


def test_completion_prompt_keeps_empty_list_on_non_chat_path() -> None:
    assert CompletionPrompt([]).to_formatted_prompt() == []


def test_direct_chat_to_text_conversion_rejects_empty_list() -> None:
    with pytest.raises(AssertionError, match="Expected a chat prompt"):
        chat_prompt_to_text_prompt([])


def test_chat_completion_prompt_rejects_empty_list_with_text_prompt_error() -> None:
    with pytest.raises(AssertionError, match="Expected a text prompt"):
        ChatCompletionPrompt([]).to_formatted_prompt()
