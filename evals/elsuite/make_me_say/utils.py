import functools
from typing import Callable, Union

import backoff
import openai
from openai import OpenAI

client = OpenAI()
import openai
import urllib3.exceptions

from evals.api import CompletionResult


@backoff.on_exception(
    backoff.expo,
    (
        openai.RateLimitError,
        urllib3.exceptions.TimeoutError,
    ),
)
def openai_chatcompletion_create(*args, **kwargs):
    return client.chat.completions.create(*args, **kwargs)


def get_completion(prompt, model_name):
    return openai_chatcompletion_create(
        model=model_name,
        messages=prompt,
    )


def get_completion_fn(model_name: str) -> Callable[[Union[str, list[dict]]], Union[str, dict]]:
    return functools.partial(get_completion, model_name=model_name)


def get_content(response: Union[dict, CompletionResult]) -> str:
    if hasattr(response, "get_completions"):
        completions = response.get_completions()
        assert len(completions) == 1, f"Got {len(completions)} but expected exactly one"
        return completions[0]

    return response.choices[0].message.content
