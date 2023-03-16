from functools import cached_property
from typing import Optional

import openai

from evals.base import ModelSpec
from evals.prompt.base import OpenAICreateChatPrompt, OpenAICreatePrompt
from evals.utils.api_utils import (
    openai_chat_completion_create_retrying,
    openai_completion_create_retrying,
)

from .base import _ModelRunner


class OpenAIRunner(_ModelRunner):
    def completion(self, prompt: OpenAICreatePrompt, **kwargs):
        return openai_completion_create_retrying(prompt=prompt, **kwargs)

    def chat_completion(self, messages: OpenAICreateChatPrompt, **kwargs):
        return openai_chat_completion_create_retrying(messages=messages, **kwargs)

    @classmethod
    def resolve(cls, name: str) -> ModelSpec:
        resolver = ModelResolver()
        return resolver.resolve(name)


def n_ctx_from_model_name(model_name: str) -> Optional[int]:
    """Returns n_ctx for a given API model name. Model list last updated 2023-03-14."""
    # note that for most models, the max tokens is n_ctx + 1
    DICT_OF_N_CTX_BY_MODEL_NAME_PREFIX: dict[str, int] = {
        "gpt-3.5-turbo-": 4096,
        "gpt-4-": 8192,
        "gpt-4-32k-": 32768,
    }
    DICT_OF_N_CTX_BY_MODEL_NAME: dict[str, int] = {
        "ada": 2048,
        "text-ada-001": 2048,
        "babbage": 2048,
        "text-babbage-001": 2048,
        "curie": 2048,
        "text-curie-001": 2048,
        "davinci": 2048,
        "text-davinci-001": 2048,
        "code-davinci-002": 8000,
        "text-davinci-002": 4096,
        "text-davinci-003": 4096,
        "gpt-3.5-turbo": 4096,
        "gpt-3.5-turbo-0301": 4096,
        "gpt-4": 8192,
        "gpt-4-0314": 8192,
        "gpt-4-32k": 32768,
        "gpt-4-32k-0314": 32768,
    }
    # first, look for a prefix match
    for model_prefix, n_ctx in DICT_OF_N_CTX_BY_MODEL_NAME_PREFIX.items():
        if model_name.startswith(model_prefix):
            return n_ctx
    # otherwise, look for an exact match and return None if not found
    return DICT_OF_N_CTX_BY_MODEL_NAME.get(model_name, None)


class ModelResolver:
    # This is a temporary method to identify which models are chat models.
    # Eventually, the OpenAI API should expose this information directly.
    CHAT_MODELS = {
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0301",
        "gpt-4",
        "gpt-4-0314",
        "gpt-4-32k",
        "gpt-4-32k-0314",
    }

    def resolve(self, name: str) -> ModelSpec:
        if name in self.api_model_ids:
            result = ModelSpec(
                driver="openai",
                name=name,
                model=name,
                is_chat=(name in self.CHAT_MODELS),
                n_ctx=n_ctx_from_model_name(name),
            )
            return result

        raise ValueError(f"Couldn't find model: {name}")

    @cached_property
    def api_model_ids(self):
        return [m["id"] for m in openai.Model.list()["data"]]
