import pytest

from evals.utils import api_utils


def test_false_result_is_not_retried() -> None:
    calls = 0

    def return_false() -> bool:
        nonlocal calls
        calls += 1
        return False

    assert api_utils.create_retrying(return_false, (RuntimeError,)) is False
    assert calls == 1


def test_non_retryable_exception_propagates() -> None:
    def fail() -> None:
        raise ValueError("invalid request")

    with pytest.raises(ValueError, match="invalid request"):
        api_utils.create_retrying(fail, (RuntimeError,))


def test_retry_configuration_uses_exception_and_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_on_exception(**config):
        captured.update(config)

        def decorate(func):
            return func

        return decorate

    monkeypatch.setattr(api_utils.backoff, "on_exception", fake_on_exception)

    assert api_utils.create_retrying(lambda: "ok", (RuntimeError,)) == "ok"
    assert captured["wait_gen"] is api_utils.backoff.expo
    assert captured["exception"] == (RuntimeError,)
    assert captured["max_value"] == 60
    assert captured["factor"] == 1.5
    assert captured["max_time"] == api_utils.EVALS_THREAD_TIMEOUT


def test_retryable_failures_really_retry_then_return_result(monkeypatch):
    import backoff._sync

    monkeypatch.setattr(backoff._sync.time, "sleep", lambda _: None)
    calls = []

    def operation(value):
        calls.append(value)
        if len(calls) < 3:
            raise RuntimeError("temporary")
        return "completed"

    assert api_utils.create_retrying(operation, (RuntimeError,), "input") == "completed"
    assert calls == ["input", "input", "input"]


def test_retry_deadline_propagates_the_original_exception(monkeypatch):
    monkeypatch.setattr(api_utils, "EVALS_THREAD_TIMEOUT", 0)
    error = RuntimeError("still unavailable")

    def operation():
        raise error

    with pytest.raises(RuntimeError) as raised:
        api_utils.create_retrying(operation, (RuntimeError,))
    assert raised.value is error


@pytest.mark.parametrize("value", [None, 0, "", []])
def test_other_falsey_success_values_do_not_retry(value):
    from unittest.mock import Mock

    operation = Mock(return_value=value)
    assert api_utils.create_retrying(operation, (RuntimeError,)) == value
    operation.assert_called_once_with()
