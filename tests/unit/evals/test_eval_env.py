import pytest

from evals.eval import _env_bool


def test_env_bool_preserves_default_when_unset(monkeypatch):
    monkeypatch.delenv("EVALS_SHOW_EVAL_PROGRESS", raising=False)

    assert _env_bool("EVALS_SHOW_EVAL_PROGRESS", True) is True
    assert _env_bool("EVALS_SHOW_EVAL_PROGRESS", False) is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "TRUE", " Yes "])
def test_env_bool_accepts_true_like_values(monkeypatch, value):
    monkeypatch.setenv("EVALS_SHOW_EVAL_PROGRESS", value)

    assert _env_bool("EVALS_SHOW_EVAL_PROGRESS", False) is True


@pytest.mark.parametrize("value", ["0", "false", "no", "FALSE", " No ", ""])
def test_env_bool_accepts_false_like_values(monkeypatch, value):
    monkeypatch.setenv("EVALS_SHOW_EVAL_PROGRESS", value)

    assert _env_bool("EVALS_SHOW_EVAL_PROGRESS", True) is False
