from pathlib import Path
from typing import Any


def _make_eval(max_questions: int, max_replies: int, monkeypatch: Any):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    from evals.api import DummyCompletionFn
    from evals.elsuite.twenty_questions.eval import TwentyQuestions

    return TwentyQuestions(
        completion_fns=[DummyCompletionFn()],
        samples_jsonl="unused",
        gamemaster_spec="dummy",
        max_questions=max_questions,
        max_replies=max_replies,
        eval_registry_path=Path("."),
    )


def test_twenty_questions_preserves_equal_max_replies(monkeypatch: Any) -> None:
    evaluator = _make_eval(max_questions=20, max_replies=20, monkeypatch=monkeypatch)

    assert evaluator.max_replies == 20


def test_twenty_questions_expands_only_lower_max_replies(monkeypatch: Any) -> None:
    evaluator = _make_eval(max_questions=20, max_replies=19, monkeypatch=monkeypatch)

    assert evaluator.max_replies == 40
