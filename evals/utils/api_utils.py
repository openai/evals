import logging
import os

import backoff

EVALS_THREAD_TIMEOUT = float(os.environ.get("EVALS_THREAD_TIMEOUT", "40"))
logging.getLogger("httpx").setLevel(logging.WARNING)  # suppress "OK" logs from openai API calls


def create_retrying(func: callable, retry_exceptions: tuple[Exception], *args, **kwargs):
    """Call ``func`` and retry only the configured exceptions."""
    retrying_func = backoff.on_exception(
        wait_gen=backoff.expo,
        exception=retry_exceptions,
        max_value=60,
        factor=1.5,
        max_time=EVALS_THREAD_TIMEOUT,
    )(func)
    return retrying_func(*args, **kwargs)
