from typing import Any


def test_reverse_roles_preserves_system_messages(monkeypatch: Any) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    from evals.elsuite.ballots.utils import reverse_roles

    messages = [
        {"role": "assistant", "content": "assistant message"},
        {"role": "user", "content": "user message"},
        {"role": "system", "content": "system instruction"},
    ]

    assert reverse_roles(messages) == [
        {"role": "user", "content": "assistant message"},
        {"role": "assistant", "content": "user message"},
        {"role": "system", "content": "system instruction"},
    ]
