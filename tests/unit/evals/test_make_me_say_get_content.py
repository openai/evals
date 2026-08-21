from typing import Any


def test_get_content_supports_dictionary_responses(monkeypatch: Any) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    from evals.elsuite.make_me_say.utils import get_content

    response = {"choices": [{"message": {"content": "hello"}}]}

    assert get_content(response) == "hello"


def test_get_content_preserves_completion_result_behavior(monkeypatch: Any) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    from evals.elsuite.make_me_say.utils import get_content

    class StaticCompletionResult:
        def get_completions(self) -> list[str]:
            return ["hello"]

    assert get_content(StaticCompletionResult()) == "hello"
