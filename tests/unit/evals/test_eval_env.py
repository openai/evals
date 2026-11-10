from contextlib import nullcontext
from importlib import import_module
from types import SimpleNamespace

import pytest

from evals.eval import Eval, SolverEval, _env_bool


@pytest.mark.parametrize(
    "flag", ["EVALS_SHOW_EVAL_PROGRESS", "EVALS_SEQUENTIAL", "EVALS_GENTLE_INTERRUPT"]
)
def test_env_bool_preserves_default_when_unset(monkeypatch, flag):
    monkeypatch.delenv(flag, raising=False)

    assert _env_bool(flag, True) is True
    assert _env_bool(flag, False) is False


@pytest.mark.parametrize(
    "flag", ["EVALS_SHOW_EVAL_PROGRESS", "EVALS_SEQUENTIAL", "EVALS_GENTLE_INTERRUPT"]
)
@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE", " Yes ", " ON "])
def test_env_bool_accepts_true_like_values(monkeypatch, flag, value):
    monkeypatch.setenv(flag, value)

    assert _env_bool(flag, False) is True


@pytest.mark.parametrize(
    "flag", ["EVALS_SHOW_EVAL_PROGRESS", "EVALS_SEQUENTIAL", "EVALS_GENTLE_INTERRUPT"]
)
@pytest.mark.parametrize(
    "value", ["0", "false", "no", "off", "FALSE", " No ", " OFF ", ""]
)
def test_env_bool_accepts_false_like_values(monkeypatch, flag, value):
    monkeypatch.setenv(flag, value)

    assert _env_bool(flag, True) is False


@pytest.mark.parametrize("method", [Eval.eval_all_samples, SolverEval.eval_all_samples])
@pytest.mark.parametrize("value, threaded", [(" ON ", False), ("OFF", True)])
def test_sequential_flag_selects_execution_path(monkeypatch, method, value, threaded):
    monkeypatch.setenv("EVALS_SEQUENTIAL", value)
    monkeypatch.setenv("EVALS_SHOW_EVAL_PROGRESS", "off")
    module = import_module("evals.eval")
    calls = []

    class Pool:
        def __init__(self, threads):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def imap_unordered(self, fn, values):
            calls.append("threaded")
            return map(fn, values)

    monkeypatch.setattr(module, "ThreadPool", Pool)
    recorder = SimpleNamespace(as_default_recorder=lambda _: nullcontext())
    eval_instance = SimpleNamespace(
        name="synthetic.test",
        seed=0,
        _solver=SimpleNamespace(copy=lambda: object()),
        eval_sample=lambda *args: "result",
    )
    assert method(eval_instance, recorder, ["sample"], show_progress=False) == [
        "result"
    ]
    assert bool(calls) is threaded


@pytest.mark.parametrize("value, gentle", [(" ON ", True), ("off", False)])
def test_solver_gentle_interrupt_flag(monkeypatch, value, gentle):
    monkeypatch.setenv("EVALS_GENTLE_INTERRUPT", value)
    monkeypatch.setenv("EVALS_SHOW_EVAL_PROGRESS", "off")
    module = import_module("evals.eval")

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(module, "tqdm", interrupt)
    recorder = SimpleNamespace(as_default_recorder=lambda _: nullcontext())
    eval_instance = SimpleNamespace(
        name="synthetic.test", seed=0, _solver=SimpleNamespace(copy=lambda: object())
    )
    if gentle:
        assert SolverEval.eval_all_samples(eval_instance, recorder, ["sample"]) == []
    else:
        with pytest.raises(KeyboardInterrupt):
            SolverEval.eval_all_samples(eval_instance, recorder, ["sample"])
