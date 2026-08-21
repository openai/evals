import logging
from typing import Any, Optional, Union

import openai
import tiktoken
from openai import OpenAI

from evals.api import CompletionFn, CompletionResult
from evals.base import CompletionFnSpec
from evals.prompt.base import (
    ChatCompletionPrompt,
    CompletionPrompt,
    OpenAICreateChatPrompt,
    OpenAICreatePrompt,
    Prompt,
)
from evals.record import record_sampling
from evals.utils.api_utils import create_retrying

OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


def _encoding_for_model(model: Optional[str]):
    if model is not None:
        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            pass
    return tiktoken.get_encoding("cl100k_base")


def _estimate_prompt_tokens(prompt: Any, model: Optional[str]) -> Optional[int]:
    """Estimate input tokens for prompt shapes supported by OpenAI completion functions."""
    encoding = _encoding_for_model(model)

    if isinstance(prompt, str):
        return len(encoding.encode(prompt))

    if isinstance(prompt, list):
        if all(isinstance(token, int) for token in prompt):
            return len(prompt)
        if all(isinstance(token, str) for token in prompt):
            return sum(len(encoding.encode(token)) for token in prompt)
        if all(isinstance(message, dict) for message in prompt):
            # OpenAI chat token accounting has varied slightly between model versions.
            # This follows the standard 3-tokens-per-message approximation and is used
            # only to reject prompts that already exceed a known context window.
            tokens = 3  # assistant reply priming
            for message in prompt:
                tokens += 3
                for key, value in message.items():
                    if not isinstance(value, str):
                        continue
                    tokens += len(encoding.encode(value))
                    if key == "name":
                        tokens += 1
            return tokens

    return None


def _validate_prompt_context(prompt: Any, model: Optional[str], n_ctx: Optional[int]) -> None:
    if n_ctx is None:
        return

    prompt_tokens = _estimate_prompt_tokens(prompt, model)
    if prompt_tokens is not None and prompt_tokens > n_ctx:
        model_name = model or "the selected model"
        raise ValueError(
            f"Prompt is approximately {prompt_tokens} tokens, exceeding {model_name}'s "
            f"configured context window of {n_ctx} tokens. Use a model with a larger "
            "context window or shorten the eval prompt."
        )


def openai_completion_create_retrying(client: OpenAI, *args, **kwargs):
    """
    Helper function for creating a completion.
    `args` and `kwargs` match what is accepted by `openai.Completion.create`.
    """
    result = create_retrying(
        client.completions.create, retry_exceptions=OPENAI_TIMEOUT_EXCEPTIONS, *args, **kwargs
    )
    if "error" in result:
        logging.warning(result)
        raise openai.APIError(result["error"])
    return result


def openai_chat_completion_create_retrying(client: OpenAI, *args, **kwargs):
    """
    Helper function for creating a completion.
    `args` and `kwargs` match what is accepted by `openai.Completion.create`.
    """
    result = create_retrying(
        client.chat.completions.create, retry_exceptions=OPENAI_TIMEOUT_EXCEPTIONS, *args, **kwargs
    )
    if "error" in result:
        logging.warning(result)
        raise openai.APIError(result["error"])
    return result


class OpenAIBaseCompletionResult(CompletionResult):
    def __init__(self, raw_data: Any, prompt: Any):
        self.raw_data = raw_data
        self.prompt = prompt

    def get_completions(self) -> list[str]:
        raise NotImplementedError


class OpenAIChatCompletionResult(OpenAIBaseCompletionResult):
    def get_completions(self) -> list[str]:
        completions = []
        if self.raw_data:
            for choice in self.raw_data.choices:
                if choice.message.content is not None:
                    completions.append(choice.message.content)
        return completions


class OpenAICompletionResult(OpenAIBaseCompletionResult):
    def get_completions(self) -> list[str]:
        completions = []
        if self.raw_data:
            for choice in self.raw_data.choices:
                completions.append(choice.text)
        return completions


class OpenAICompletionFn(CompletionFn):
    def __init__(
        self,
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        n_ctx: Optional[int] = None,
        extra_options: Optional[dict] = {},
        **kwargs,
    ):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.n_ctx = n_ctx
        self.extra_options = extra_options

    def __call__(
        self,
        prompt: Union[str, OpenAICreateChatPrompt],
        **kwargs,
    ) -> OpenAICompletionResult:
        if not isinstance(prompt, Prompt):
            assert (
                isinstance(prompt, str)
                or (isinstance(prompt, list) and all(isinstance(token, int) for token in prompt))
                or (isinstance(prompt, list) and all(isinstance(token, str) for token in prompt))
                or (isinstance(prompt, list) and all(isinstance(msg, dict) for msg in prompt))
            ), f"Got type {type(prompt)}, with val {type(prompt[0])} for prompt, expected str or list[int] or list[str] or list[dict[str, str]]"

            prompt = CompletionPrompt(
                raw_prompt=prompt,
            )

        openai_create_prompt: OpenAICreatePrompt = prompt.to_formatted_prompt()
        _validate_prompt_context(openai_create_prompt, self.model, self.n_ctx)

        result = openai_completion_create_retrying(
            OpenAI(api_key=self.api_key, base_url=self.api_base),
            model=self.model,
            prompt=openai_create_prompt,
            **{**kwargs, **self.extra_options},
        )
        result = OpenAICompletionResult(raw_data=result, prompt=openai_create_prompt)
        record_sampling(
            prompt=result.prompt,
            sampled=result.get_completions(),
            model=result.raw_data.model,
            usage=result.raw_data.usage,
        )
        return result


class OpenAIChatCompletionFn(CompletionFnSpec):
    def __init__(
        self,
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        n_ctx: Optional[int] = None,
        extra_options: Optional[dict] = {},
    ):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.n_ctx = n_ctx
        self.extra_options = extra_options

    def __call__(
        self,
        prompt: Union[str, OpenAICreateChatPrompt],
        **kwargs,
    ) -> OpenAIChatCompletionResult:
        if not isinstance(prompt, Prompt):
            assert (
                isinstance(prompt, str)
                or (isinstance(prompt, list) and all(isinstance(token, int) for token in prompt))
                or (isinstance(prompt, list) and all(isinstance(token, str) for token in prompt))
                or (isinstance(prompt, list) and all(isinstance(msg, dict) for msg in prompt))
            ), f"Got type {type(prompt)}, with val {type(prompt[0])} for prompt, expected str or list[int] or list[str] or list[dict[str, str]]"

            prompt = ChatCompletionPrompt(
                raw_prompt=prompt,
            )

        openai_create_prompt: OpenAICreateChatPrompt = prompt.to_formatted_prompt()
        _validate_prompt_context(openai_create_prompt, self.model, self.n_ctx)

        result = openai_chat_completion_create_retrying(
            OpenAI(api_key=self.api_key, base_url=self.api_base),
            model=self.model,
            messages=openai_create_prompt,
            **{**kwargs, **self.extra_options},
        )
        result = OpenAIChatCompletionResult(raw_data=result, prompt=openai_create_prompt)
        record_sampling(
            prompt=result.prompt,
            sampled=result.get_completions(),
            model=result.raw_data.model,
            usage=result.raw_data.usage,
        )
        return result
