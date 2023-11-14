"""
This file defines various helper functions for interacting with the OpenAI API.
"""
import concurrent
import logging
import os

import backoff
from openai import OpenAI
import openai

client = OpenAI()

EVALS_THREAD_TIMEOUT = float(os.environ.get("EVALS_THREAD_TIMEOUT", "40"))


@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=(
        openai.APIError,
        openai.APIStatusError,
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.APIConnectionError,
        openai.InternalServerError,
    ),
    max_value=60,
    factor=1.5,
)
def openai_completion_create_retrying(*args, **kwargs):
    """
    Helper function for creating a completion.
    `args` and `kwargs` match what is accepted by `openai.Completion.create`.
    """
    if "api_base" in kwargs:
        del kwargs["api_base"]
    if "api_key" in kwargs:
        del kwargs["api_key"]
    result = client.completions.create(*args, **kwargs)
    if "error" in result:
        logging.warning(result)
        raise openai.APIError(result["error"])
    return result


def request_with_timeout(func, *args, timeout=EVALS_THREAD_TIMEOUT, **kwargs):
    """
    Worker thread for making a single request within allotted time.
    """
    while True:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                result = future.result(timeout=timeout)
                return result
            except concurrent.futures.TimeoutError:
                continue


@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=(
        openai.APIError,
        openai.APIStatusError,
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.APIConnectionError,
        openai.InternalServerError,
    ),
    max_value=60,
    factor=1.5,
)
def openai_chat_completion_create_retrying(*args, **kwargs):
    """
    Helper function for creating a chat completion.
    `args` and `kwargs` match what is accepted by `openai.ChatCompletion.create`.
    """
    if "api_base" in kwargs:
        del kwargs["api_base"]
    if "api_key" in kwargs:
        del kwargs["api_key"]
    result = request_with_timeout(openai.chat.completions.create, *args, **kwargs)
    if "error" in result:
        logging.warning(result)
        raise openai.APIError(result["error"])
    return result
