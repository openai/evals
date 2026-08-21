from typing import Any

import pytest


class StaticCompletionResult:
    def __init__(self, completion: str) -> None:
        self.completion = completion

    def get_completions(self) -> list[str]:
        return [self.completion]


class StaticCompletionFn:
    def __init__(self, completion: str) -> None:
        self.completion = completion

    def __call__(self, prompt: str) -> StaticCompletionResult:
        return StaticCompletionResult(self.completion)


@pytest.mark.parametrize(
    "completion, expected, exact, fuzzy",
    [
        ("", "answer", 0, 0),
        ("answer", "", 0, 0),
        ("", "", 1, 0),
        ("answer", "the answer is here", 0, 1),
        ("the answer is here", "answer", 0, 1),
    ],
)
def test_self_prompting_empty_outputs_do_not_fuzzy_match(
    completion: str,
    expected: str,
    exact: int,
    fuzzy: int,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    from evals.elsuite.self_prompting.eval import SelfPrompting
    from evals.record import DummyRecorder

    evaluator = object.__new__(SelfPrompting)
    evaluator.tasker_completion_fns = {"dummy": StaticCompletionFn(completion)}
    recorder = DummyRecorder(None)
    sample = {
        "tasker_model": "dummy",
        "model_instruction": "Respond to this input:",
        "input": "question",
        "output": expected,
    }

    with recorder.as_default_recorder("sample"):
        result = evaluator._run_tasking(sample)

    assert result["exact"] == exact
    assert result["fuzzy"] == fuzzy
