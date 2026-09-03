import importlib.util
import time
from pathlib import Path
from unittest.mock import Mock

import pytest


def _load_api_utils():
    path = Path(__file__).resolve().parents[1] / "evals" / "utils" / "api_utils.py"
    spec = importlib.util.spec_from_file_location("evals_api_utils", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_create_retrying_gives_up_instead_of_looping_forever(monkeypatch):
    api_utils = _load_api_utils()
    monkeypatch.setattr(api_utils, "EVALS_API_RETRY_MAX_TIME", 0)
    error = ConnectionError("boom")
    # A second call raises a different exception, so a broken retry bound
    # fails immediately instead of leaving the regression test in a loop.
    func = Mock(side_effect=[error, RuntimeError("unexpected retry")])
    with pytest.raises(ConnectionError) as exc:
        api_utils.create_retrying(func, retry_exceptions=(ConnectionError,))
    assert exc.value is error
    func.assert_called_once_with()


@pytest.mark.parametrize("result", [None, False, [], {}, "ok"])
def test_successful_results_are_not_retried(monkeypatch, result):
    api_utils = _load_api_utils()
    monkeypatch.setattr(time, "sleep", lambda _: None)
    func = Mock(side_effect=[result, RuntimeError("unexpected retry")])
    assert api_utils.create_retrying(func, (ConnectionError,), "input", option=1) is result
    func.assert_called_once_with("input", option=1)


def test_retryable_failure_can_recover(monkeypatch):
    api_utils = _load_api_utils()
    monkeypatch.setattr(time, "sleep", lambda _: None)
    func = Mock(side_effect=[ConnectionError("retry"), "ok"])
    assert api_utils.create_retrying(func, (ConnectionError,), option=1) == "ok"
    assert func.call_count == 2
    func.assert_called_with(option=1)


def test_non_retryable_failure_propagates():
    api_utils = _load_api_utils()
    error = ValueError("invalid request")
    func = Mock(side_effect=error)
    with pytest.raises(ValueError) as exc:
        api_utils.create_retrying(func, (ConnectionError,))
    assert exc.value is error
    func.assert_called_once_with()


def test_retry_budget_does_not_cancel_an_in_flight_call(monkeypatch):
    api_utils = _load_api_utils()
    monkeypatch.setattr(api_utils, "EVALS_API_RETRY_MAX_TIME", 0.001)

    def slow_success():
        time.sleep(0.02)
        return "ok"

    assert api_utils.create_retrying(slow_success, (ConnectionError,)) == "ok"
