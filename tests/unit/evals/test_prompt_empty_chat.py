from typing import Any

import pytest


def test_empty_list_is_not_a_chat_prompt(monkeypatch: Any) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    from evals.prompt.base import (
        CompletionPrompt,
        chat_prompt_to_text_prompt,
        is_chat_prompt,
    )

    assert not is_chat_prompt([])
    assert CompletionPrompt([]).to_formatted_prompt() == []
    with pytest.raises(AssertionError, match="Expected a chat prompt"):
        chat_prompt_to_text_prompt([])


def test_nonempty_message_list_is_still_a_chat_prompt(monkeypatch: Any) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    from evals.prompt.base import is_chat_prompt

    prompt = [{"role": "user", "content": "hello"}]

    assert is_chat_prompt(prompt)
    assert not is_chat_prompt(["hello"])
