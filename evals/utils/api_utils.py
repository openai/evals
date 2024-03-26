import logging
import os

import backoff

EVALS_THREAD_TIMEOUT = float(os.environ.get("EVALS_THREAD_TIMEOUT", "40"))
logging.getLogger("httpx").setLevel(logging.WARNING)  # suppress "OK" logs from openai API calls


@backoff.on_predicate(
    wait_gen=backoff.expo,
    max_value=60,
    factor=1.5,
)
def create_retrying(func: callable, retry_exceptions: tuple[Exception], *args, **kwargs):
    """
    Retries given function if one of given exceptions is raised
    """
    try:
        return func(*args, **kwargs)
    except retry_exceptions:
        return False
