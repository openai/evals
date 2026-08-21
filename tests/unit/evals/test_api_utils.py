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
