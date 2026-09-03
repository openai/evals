import pytest

from evals.utils import api_utils


def test_create_retrying_re_raises_after_retry_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    class RetryableError(Exception):
        pass

    attempts = 0

    def always_fail() -> None:
        nonlocal attempts
        attempts += 1
        raise RetryableError("provider unavailable")

    monkeypatch.setattr(api_utils, "EVALS_API_RETRY_MAX_TRIES", 1)

    with pytest.raises(RetryableError, match="provider unavailable"):
        api_utils.create_retrying(always_fail, (RetryableError,))

    assert attempts == 1


def test_create_retrying_returns_success_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_utils, "EVALS_API_RETRY_MAX_TRIES", 1)

    assert api_utils.create_retrying(lambda: "ok", (RuntimeError,)) == "ok"
