import logging
import os

import backoff

EVALS_THREAD_TIMEOUT = float(os.environ.get("EVALS_THREAD_TIMEOUT", "40"))
EVALS_API_RETRY_MAX_TRIES = int(os.environ.get("EVALS_API_RETRY_MAX_TRIES", "8"))
logging.getLogger("httpx").setLevel(logging.WARNING)  # suppress "OK" logs from openai API calls


def create_retrying(func: callable, retry_exceptions: tuple[Exception], *args, **kwargs):
    """
    Retries given function if one of given exceptions is raised.

    Retry attempts are bounded so persistent provider outages surface an exception
    instead of blocking an evaluation indefinitely.
    """

    @backoff.on_exception(
        wait_gen=backoff.expo,
        exception=retry_exceptions,
        max_value=60,
        factor=1.5,
        max_tries=EVALS_API_RETRY_MAX_TRIES,
    )
    def call():
        return func(*args, **kwargs)

    return call()
