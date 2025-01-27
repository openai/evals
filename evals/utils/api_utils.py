import logging
import os
import backoff
import requests

from openai import APIError, APIConnectionError, APITimeoutError, RateLimitError

EVALS_THREAD_TIMEOUT = float(os.environ.get("EVALS_THREAD_TIMEOUT", "40"))
logging.getLogger("httpx").setLevel(logging.WARNING)  # Suppress "OK" logs from OpenAI API

RETRY_ERRORS = (
    APIConnectionError,
    APIError,
    APITimeoutError,
    RateLimitError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
)

@backoff.on_predicate(
    wait_gen=backoff.expo,
    max_value=60,
    factor=1.5,
)
def create_retrying(func: callable, retry_exceptions: tuple[Exception] = RETRY_ERRORS, *args, **kwargs):
    """
    Retries given function if one of given exceptions is raised
    """
    try:
        return func(*args, **kwargs)
    except retry_exceptions as e:
        logging.warning(f"Retrying due to error: {str(e)}")
        return False