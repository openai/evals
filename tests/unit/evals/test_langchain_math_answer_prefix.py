from typing import Any

import pytest


class StaticMathChain:
    def __init__(self, response: str) -> None:
        self.response = response

    def run(self, prompt: str) -> str:
        return self.response


@pytest.mark.parametrize(
    "response, expected",
    [
        ("Answer: ten", "ten"),
        ("Answer: seven", "seven"),
        ("Answer: 42", "42"),
        ("ten", "ten"),
    ],
)
def test_langchain_math_removes_only_exact_answer_prefix(
    response: str, expected: str, monkeypatch: Any
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    from evals.completion_fns import langchain_math

    completion_fn = object.__new__(langchain_math.LangChainMathChainCompletionFn)
    completion_fn.llm_math = StaticMathChain(response)
    monkeypatch.setattr(langchain_math, "record_sampling", lambda **kwargs: None)

    result = completion_fn("question")

    assert result.get_completions() == [expected]
