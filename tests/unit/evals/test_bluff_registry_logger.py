from typing import Any

import pytest


@pytest.mark.parametrize("initially_disabled", [False, True])
def test_bluff_solver_lookup_restores_registry_logger(
    initially_disabled: bool, monkeypatch: Any
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    from evals.elsuite.bluff import eval as bluff_eval

    def fail_lookup(_solver_name: str):
        raise ValueError("not a registered solver")

    monkeypatch.setattr(bluff_eval.registry, "make_completion_fn", fail_lookup)
    monkeypatch.setattr(bluff_eval.evals.registry.logger, "disabled", initially_disabled)

    with pytest.raises(ValueError, match="not a registered solver"):
        bluff_eval.BluffEval._create_solver_player(object(), "missing-solver")

    assert bluff_eval.evals.registry.logger.disabled is initially_disabled
