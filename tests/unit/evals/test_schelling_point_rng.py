import random
from pathlib import Path
from typing import Any


def _make_eval(monkeypatch: Any, seed: int):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    from evals.api import DummyCompletionFn
    from evals.elsuite.schelling_point.eval import SchellingPoint

    return SchellingPoint(
        completion_fns=[DummyCompletionFn()],
        seed=seed,
        eval_registry_path=Path("."),
    )


def test_schelling_point_does_not_reseed_global_rng(monkeypatch: Any) -> None:
    random.seed(8675309)
    global_state = random.getstate()

    evaluator = _make_eval(monkeypatch, seed=42)

    assert random.getstate() == global_state
    assert evaluator.seed == 42


def test_schelling_point_sample_uses_supplied_rng(monkeypatch: Any) -> None:
    from evals.elsuite.schelling_point import eval as schelling_eval

    evaluator = _make_eval(monkeypatch, seed=123)
    monkeypatch.setattr(schelling_eval, "get_response", lambda *_args: ("same", "scratchpad"))
    monkeypatch.setattr(schelling_eval.evals.record, "record_metrics", lambda **_kwargs: None)

    random.seed(314159)
    global_state = random.getstate()
    evaluator.eval_sample({"0": "first", "1": "second"}, random.Random(123))

    assert random.getstate() == global_state
