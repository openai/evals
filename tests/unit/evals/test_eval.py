import pytest

from evals.eval import _show_progress_from_env


@pytest.fixture
def clear_show_progress_env(monkeypatch):
    monkeypatch.delenv("EVALS_SHOW_EVAL_PROGRESS", raising=False)


def test_show_progress_unset_returns_default(clear_show_progress_env):
    assert _show_progress_from_env(True) is True
    assert _show_progress_from_env(False) is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "TRUE", "Yes"])
def test_show_progress_truthy_strings(monkeypatch, value):
    monkeypatch.setenv("EVALS_SHOW_EVAL_PROGRESS", value)
    assert _show_progress_from_env(False) is True


@pytest.mark.parametrize("value", ["0", "false", "no", "", "off"])
def test_show_progress_falsy_strings(monkeypatch, value):
    """`bool("0")` is True in Python, which is the original bug."""
    monkeypatch.setenv("EVALS_SHOW_EVAL_PROGRESS", value)
    assert _show_progress_from_env(True) is False
