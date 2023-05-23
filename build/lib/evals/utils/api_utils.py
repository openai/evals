"""
This file defines various helper functions for interacting with the OpenAI API.
"""
import logging

import backoff
import openai


def generate_dummy_chat_completion():
    return {
        "id": "dummy-id",
        "object": "chat.completion",
        "created": 12345,
        "model": "dummy-chat",
        "usage": {"prompt_tokens": 56, "completion_tokens": 6, "total_tokens": 62},
        "choices": [
            {
                "message": {"role": "assistant", "content": "This is a dummy response."},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
    }


def generate_dummy_completion():
    return {
        "id": "dummy-id",
        "object": "text_completion",
        "created": 12345,
        "model": "dummy-completion",
        "choices": [
            {
                "text": "This is a dummy response.",
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
    }


@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=(
        openai.error.ServiceUnavailableError,
        openai.error.APIError,
        openai.error.RateLimitError,
        openai.error.APIConnectionError,
        openai.error.Timeout,
    ),
    max_value=60,
    factor=1.5,
)
def openai_completion_create_retrying(*args, **kwargs):
    """
    Helper function for creating a completion.
    `args` and `kwargs` match what is accepted by `openai.Completion.create`.
    """
    if kwargs["model"] == "dummy-completion":
        return generate_dummy_completion()

    result = openai.Completion.create(*args, **kwargs)
    if "error" in result:
        logging.warning(result)
        raise openai.error.APIError(result["error"])
    return result


@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=(
        openai.error.ServiceUnavailableError,
        openai.error.APIError,
        openai.error.RateLimitError,
        openai.error.APIConnectionError,
        openai.error.Timeout,
    ),
    max_value=60,
    factor=1.5,
)
def openai_chat_completion_create_retrying(*args, **kwargs):
    """
    Helper function for creating a chat completion.
    `args` and `kwargs` match what is accepted by `openai.ChatCompletion.create`.
    """
    if kwargs["model"] == "dummy-chat":
        return generate_dummy_chat_completion()

    result = openai.ChatCompletion.create(*args, **kwargs)
    if "error" in result:
        logging.warning(result)
        raise openai.error.APIError(result["error"])
    return result
