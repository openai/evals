from unittest.mock import MagicMock

import pytest


@pytest.fixture
def eval_module(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-offline-validation")
    import evals.eval

    return evals.eval


def _args(max_samples):
    from evals.cli.oaieval import OaiEvalArguments

    args = OaiEvalArguments()
    args.debug = False
    args.visible = None
    args.max_samples = max_samples
    args.registry_path = []
    args.eval = "missing-eval"
    return args


def test_run_without_max_samples_clears_previous_in_process_limit(monkeypatch, eval_module) -> None:
    from evals.cli.oaieval import run

    monkeypatch.setattr(eval_module, "_MAX_SAMPLES", 1)

    registry = MagicMock()
    registry.get_eval.return_value = None
    registry._evals = {}

    # The missing eval stops the run immediately after per-run setup. That is
    # enough to verify that an uncapped invocation clears stale global state.
    with pytest.raises(AssertionError, match="Eval missing-eval not found"):
        run(_args(None), registry=registry)

    assert eval_module._MAX_SAMPLES is None


def test_run_sets_current_max_samples_before_registry_lookup(monkeypatch, eval_module) -> None:
    from evals.cli.oaieval import run

    monkeypatch.setattr(eval_module, "_MAX_SAMPLES", None)

    registry = MagicMock()
    registry.get_eval.return_value = None
    registry._evals = {}

    with pytest.raises(AssertionError, match="Eval missing-eval not found"):
        run(_args(7), registry=registry)

    assert eval_module._MAX_SAMPLES == 7
