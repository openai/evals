import random
from typing import Any, Optional


def _get_samples(n_samples: Optional[int], monkeypatch: Any) -> list[dict]:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    from evals.elsuite.already_said_that.eval import AlreadySaidThat

    evaluator = object.__new__(AlreadySaidThat)
    evaluator.n_samples = n_samples
    evaluator.rng = random.Random(0)
    source = [{"id": i} for i in range(4)]
    evaluator.get_samples = lambda: source
    return evaluator._get_samples()


def test_already_said_that_none_uses_all_samples(monkeypatch: Any) -> None:
    samples = _get_samples(None, monkeypatch)

    assert len(samples) == 4
    assert {sample["id"] for sample in samples} == {0, 1, 2, 3}


def test_already_said_that_explicit_cap_is_preserved(monkeypatch: Any) -> None:
    samples = _get_samples(2, monkeypatch)

    assert len(samples) == 2
