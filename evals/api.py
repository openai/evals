"""
This file provides common interfaces and utilities used by eval creators to
sample from models and process the results.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Protocol, Union, runtime_checkable

from evals.prompt.base import (
    ChatCompletionPrompt,
    CompletionPrompt,
    OpenAICreateChatPrompt,
    OpenAICreatePrompt,
    Prompt,
)
from evals.record import record_match, record_sampling
from evals.utils.api_utils import (
    openai_chat_completion_create_retrying,
    openai_completion_create_retrying,
)

logger = logging.getLogger(__name__)


class CompletionResult(ABC):
    @abstractmethod
    def get_completions(self) -> list[str]:
        pass


@runtime_checkable
class CompletionFn(Protocol):
    def __call__(
        self,
        prompt: Union[str, OpenAICreateChatPrompt],
        **kwargs,
    ) -> CompletionResult:
        """
        ARGS
        ====
        `prompt`: Either a `Prompt` object or a raw prompt that will get wrapped in
            the approriate `Prompt` class.
        `kwargs`: Other arguments passed to the API.

        RETURNS
        =======
        The result of the API call.
        The prompt that was fed into the API call as a str.
        """


class OpenAICompletionResult(CompletionResult):
    def __init__(self, raw_data: Any, prompt: Any):
        self.raw_data = raw_data
        self.prompt = prompt

    def get_completions(self) -> list[str]:
        raise NotImplementedError


class OpenAIChatCompletionResult(OpenAICompletionResult):
    def get_completions(self) -> list[str]:
        completions = []
        if self.raw_data and "choices" in self.raw_data:
            for choice in self.raw_data["choices"]:
                if "message" in choice:
                    completions.append(choice["message"]["content"])
        return completions


class OpenAICompletionResult(OpenAICompletionResult):
    def get_completions(self) -> list[str]:
        completions = []
        if self.raw_data and "choices" in self.raw_data:
            for choice in self.raw_data["choices"]:
                if "text" in choice:
                    completions.append(choice["text"])
        return completions


class OpenAICompletionFn(CompletionFn):
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
        prompt: Union[OpenAICreatePrompt, Prompt],
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

        result = openai_completion_create_retrying(
            model=self.model,
            api_base=self.api_base,
            api_key=self.api_key,
            prompt=openai_create_prompt,
            **{**kwargs, **self.extra_options},
        )
        result = OpenAICompletionResult(raw_data=result, prompt=openai_create_prompt)
        record_sampling(prompt=result.prompt, sampled=result.get_completions())
        return result


class OpenAIChatCompletionFn(CompletionFn):
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
        prompt: Union[OpenAICreateChatPrompt, Prompt],
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

        result = openai_chat_completion_create_retrying(
            model=self.model,
            api_base=self.api_base,
            api_key=self.api_key,
            messages=openai_create_prompt,
            **{**kwargs, **self.extra_options},
        )
        result = OpenAIChatCompletionResult(raw_data=result, prompt=openai_create_prompt)
        record_sampling(prompt=result.prompt, sampled=result.get_completions())
        return result


class DummyCompletionResult(CompletionResult):
    def get_completions(self) -> list[str]:
        return ["This is a dummy response."]


class DummyCompletionFn(CompletionFn):
    def __call__(
        self, prompt: Union[OpenAICreatePrompt, OpenAICreateChatPrompt, Prompt], **kwargs
    ) -> CompletionResult:
        return DummyCompletionResult()


def record_and_check_match(
    prompt: Any,
    sampled: str,
    expected: Union[str, list[str], tuple[str]],
    separator: Callable[[str], bool] = None,
    options: Optional[list[str]] = None,
):
    """
    Records and checks if a sampled response from a CompletionFn matches the expected result.

    Args:
        prompt: The input prompt.
        sampled: The sampled response from the model.
        expected: The expected response or list of responses.
        separator: Optional function to check if a character is a separator.
        options: Optional list of options to match against the sampled response.

    Returns:
        The matched option or None if no match found.
    """
    if isinstance(expected, tuple):
        expected = list(expected)
    elif not isinstance(expected, list):
        expected = [expected]
    if options is None:
        options = expected

    picked = None
    for option in options:
        if not sampled.startswith(option):
            continue
        if (
            separator is not None
            and len(sampled) > len(option)
            and not separator(sampled[len(option)])
        ):
            continue
        picked = option
        break

    result = {
        "prompt": prompt,
        "sampled": sampled,
        "options": options,
        "picked": picked,
    }
    match = picked in expected
    result["expected"] = expected
    result["match"] = match
    record_match(match, expected=expected, picked=picked, sampled=sampled, options=options)
    return picked
