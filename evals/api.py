"""
This file provides common interfaces and utilities used by eval creators to
sample from models and process the results.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Protocol, Union

from evals.base import ModelSpec
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
    def __init__(self, raw_data: Any):
        self.raw_data = raw_data

    @abstractmethod
    def get_completions(self) -> list[str]:
        pass


class CompletionFn(Protocol):
    def __call__(
        self,
        model_spec: ModelSpec,
        prompt: Union[OpenAICreatePrompt, OpenAICreateChatPrompt, Prompt],
        **kwargs,
    ) -> CompletionResult:
        """
        ARGS
        ====
        `model_spec`: `ModelSpec` containing model details to use in the query.
            This should be the dict returned by `registry.get_model()`.
            If `model_spec` is not provided, we use the default model that was
                intialized at the beginning of the run.
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
        super().__init__(raw_data)
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
    def __call__(
        self,
        model_spec: ModelSpec,
        prompt: Union[OpenAICreatePrompt, Prompt],
        **kwargs,
    ) -> OpenAICompletionResult:
        if not isinstance(prompt, Prompt):
            assert (
                isinstance(prompt, str)
                or (isinstance(prompt, list) and all(isinstance(token, int) for token in prompt))
                or (isinstance(prompt, list) and all(isinstance(token, str) for token in prompt))
            ), f"Got type {type(prompt)}, with val {type(prompt[0])} for prompt, expected str or list[int] or list[str]"

            prompt = CompletionPrompt(
                raw_prompt=prompt,
            )

        openai_create_prompt: OpenAICreatePrompt = prompt.to_openai_create_prompt()

        result = openai_completion_create_retrying(
            model=model_spec.model,
            api_base=model_spec.api_base,
            api_key=model_spec.api_key,
            prompt=openai_create_prompt,
            **{**kwargs, **model_spec.extra_options},
        )
        result = OpenAICompletionResult(raw_data=result, prompt=openai_create_prompt)
        record_sampling(prompt=result.prompt, sampled=result.get_completions())
        return result


class OpenAIChatCompletionFn(CompletionFn):
    def __call__(
        self,
        model_spec: ModelSpec,
        prompt: Union[OpenAICreateChatPrompt, Prompt],
        **kwargs,
    ) -> OpenAIChatCompletionResult:
        if not isinstance(prompt, Prompt):
            assert isinstance(prompt, list) and all(
                isinstance(msg, dict) for msg in prompt
            ), f"Got type {type(prompt)}, with val {type(prompt[0])} for prompt, expected list[dict[str, str]]"

            prompt = ChatCompletionPrompt(
                raw_prompt=prompt,
            )

        openai_create_prompt: OpenAICreateChatPrompt = prompt.to_openai_create_prompt()

        result = openai_chat_completion_create_retrying(
            model=model_spec.model,
            api_base=model_spec.api_base,
            api_key=model_spec.api_key,
            messages=openai_create_prompt,
            **{**kwargs, **model_spec.extra_options},
        )
        result = OpenAIChatCompletionResult(raw_data=result, prompt=openai_create_prompt)
        record_sampling(prompt=result.prompt, sampled=result.get_completions())
        return result


def record_and_check_match(
    sampled: str,
    expected: Union[str, list[str], tuple[str]],
    separator: Callable[[str], bool] = None,
    options: Optional[list[str]] = None,
):
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
        "sampled": sampled,
        "options": options,
        "picked": picked,
    }
    match = picked in expected
    result["expected"] = expected
    result["match"] = match
    record_sampling(**result)
    record_match(match, expected=expected, picked=picked, sampled=sampled, options=options)
    return picked


def sample_freeform(
    model_spec: ModelSpec,
    prompt: Union[OpenAICreatePrompt, OpenAICreateChatPrompt, Prompt],
    *,
    completion_fn: CompletionFn = OpenAICompletionFn(),
    temperature: float = 1.0,
    top_p: float = 0.9,
    max_tokens: int = 512,
    stop: Optional[str] = None,
    n_samples: Optional[int] = None,
    **kwargs,
) -> Union[str, list[str], dict]:
    """
    Samples a freeform response from the specified model, records the sampling,
        and returns the sampled text.

    ARGS
    ====
    `model_spec`: See `completion_query`.
    `prompt`: See `completion_query`.
    `temperature`: Passed to `openai.Completion.create`.
    `top_p`: Passed to `openai.Completion.create`.
    `max_tokens`: Passed to `openai.Completion.create`.
    `stop`: Passed to `openai.Completion.create`.
    `n_samples`: The number of samples to generate (1 if None).
    `kwargs`: See `completion_query`.

    RETURNS
    =======
    Returns the sampled text, or a list of sampled texts if
        `n_samples` is not None.
    """
    result = completion_fn(
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=stop,
        n=(1 if n_samples is None else n_samples),
        model_spec=model_spec,
        headers={},
        **kwargs,
    )
    sampled = result.get_completions()[0]
    return sampled
