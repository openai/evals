import logging
import os

import backoff

EVALS_THREAD_TIMEOUT = float(os.environ.get("EVALS_THREAD_TIMEOUT", "40"))
EVALS_API_RETRY_MAX_TIME = float(os.environ.get("EVALS_API_RETRY_MAX_TIME", "300"))
logging.getLogger("httpx").setLevel(logging.WARNING)  # suppress "OK" logs from openai API calls


def create_retrying(func: callable, retry_exceptions: tuple[Exception], *args, **kwargs):
    """
    Retries given function if one of given exceptions is raised.

    Bounds retries with EVALS_API_RETRY_MAX_TIME (default 300 seconds),
    re-raising the last retryable exception when the budget is exhausted.
    This does not interrupt an in-flight call; configure request timeouts
    separately on the underlying client.
    """

    @backoff.on_exception(
        backoff.expo,
        retry_exceptions,
        max_value=60,
        factor=1.5,
        max_time=EVALS_API_RETRY_MAX_TIME,
    )
    def _call():
        return func(*args, **kwargs)

    return _call()
