"""
This file provides common interfaces and utilities used by eval creators to
sample from models and process the results.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Protocol, Union

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


class ReturnType(ABC):
    def __init__(self, raw_data: Any, prompt: Any, metadata: Dict[str, Any] = None):
        self.raw_data = raw_data
        self.prompt: str = prompt
        self.completions: List[str] = self.extract_completions()
        self.metadata = metadata if metadata is not None else {}

    @abstractmethod
    def extract_completions(self) -> List[str]:
        pass


class CompletionFn(Protocol):
    def __call__(
        self,
        model_spec: ModelSpec,
        prompt: Union[OpenAICreatePrompt, OpenAICreateChatPrompt, Prompt],
        **kwargs,
    ) -> ReturnType:
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
        A dict containing metadata about the query.
        """


class OpenAIReturnType(ReturnType):
    def __init__(self, raw_data: Any, prompt: Any, metadata: Dict[str, Any] = None):
        super().__init__(raw_data, prompt, metadata)

    def extract_completions(self) -> List[str]:
        completions = []
        if self.raw_data:
            if "choices" in self.raw_data:
                for choice in self.raw_data["choices"]:
                    if "message" in choice:  # chat response
                        completions.append(choice["message"]["content"])
                    else:  # non-chat response
                        completions.append(choice["text"])
        return completions


class OpenAICompletionFn(CompletionFn):
    def __call__(
        self,
        model_spec: ModelSpec,
        prompt: Union[OpenAICreatePrompt, OpenAICreateChatPrompt, Prompt],
        **kwargs,
    ) -> OpenAIReturnType:
        """
        Query the API for a completion.

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
        A dict containing metadata about the query.
        """
        if not isinstance(prompt, Prompt):
            assert (
                isinstance(prompt, str)
                or (isinstance(prompt, list) and all(isinstance(token, int) for token in prompt))
                or (isinstance(prompt, list) and all(isinstance(token, str) for token in prompt))
                or (isinstance(prompt, list) and all(isinstance(msg, dict) for msg in prompt))
            ), f"Got type {type(prompt)}, with val {type(prompt[0])} for prompt, expected str or list[int] or list[str] or list[dict[str, str]]"

            if model_spec.is_chat:
                prompt = ChatCompletionPrompt(
                    raw_prompt=prompt,
                )
            else:
                prompt = CompletionPrompt(
                    raw_prompt=prompt,
                )

        openai_create_prompt: Union[
            OpenAICreatePrompt, OpenAICreateChatPrompt
        ] = prompt.to_openai_create_prompt()

        if model_spec.is_chat:
            result = openai_chat_completion_create_retrying(
                model=model_spec.model,
                api_base=model_spec.api_base,
                api_key=model_spec.api_key,
                messages=openai_create_prompt,
                **{**kwargs, **model_spec.extra_options},
            )
        else:
            result = openai_completion_create_retrying(
                model=model_spec.model,
                api_base=model_spec.api_base,
                api_key=model_spec.api_key,
                prompt=openai_create_prompt,
                **{**kwargs, **model_spec.extra_options},
            )

        metadata = {}

        if result:
            metadata["completion_id"] = result.get("id", None)
            metadata["model"] = result.get("model", None)

        return OpenAIReturnType(raw_data=result, prompt=openai_create_prompt, metadata=metadata)


# TODO(hwc): remove this
def check_sampled_text(
    model_spec: ModelSpec,
    prompt: Union[OpenAICreatePrompt, OpenAICreateChatPrompt, Prompt],
    expected: Union[str, list[str], tuple[str]],
    completion_fn: CompletionFn = OpenAICompletionFn(),
    *,
    options: Optional[list[str]] = None,
    separator: Callable[[str], bool] = None,
) -> Optional[str]:
    """
    Generates a completion using the prompt, checks whether the completion is
        one of the expected completions, and then records the result.

    ARGS
    ====
    `model_spec`: See `completion_query`.
    `prompt`: See `completion_query`.
    `options`: The list of canonical options, defaults to `expected` if None.
        The completion will be converted to one of these options.
    `expected`: The desired completion or the list of desired completions.
    `separator`: A callable which check the character sampled after the option
        to see if it is a valid separator.

    RETURNS
    =======
    The option that was picked, i.e., matched the completion, or None.
    """
    result = completion_fn(
        prompt=prompt,
        model_spec=model_spec,
    )

    completion = result.extract_completions()[0]
    sampled = completion.strip() if model_spec.strip_completion else completion

    return record_and_check_match(
        prompt=result.prompt,
        sampled=sampled,
        expected=expected,
        metadata=result.metadata,
        separator=separator,
        options=options,
    )


def record_and_check_match(
    prompt: Union[OpenAICreatePrompt, OpenAICreateChatPrompt],
    sampled: str,
    expected: Union[str, list[str], tuple[str]],
    metadata: dict,
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
        "prompt": prompt,
        "sampled": sampled,
        "options": options,
        "picked": picked,
    }
    match = picked in expected
    result["expected"] = expected
    result["match"] = match
    result["metadata"] = metadata
    record_sampling(**result)
    record_match(match, expected=expected, picked=picked, sampled=sampled)
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
    return_logprobs: bool = False,
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
    `return_logprobs`: If True, returns the tokens and corresponding logprobs
        in addition to the sampled text.
    `kwargs`: See `completion_query`.

    RETURNS
    =======
    If `return_logprobs` is True, returns a dict with the sampled text, tokens,
        and corresponding logprobs. If `n_samples` is None, the outer list is
        removed from all values.
    Otherwise, returns the sampled text, or a list of sampled texts if
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

    return postprocess_sample_freeform(
        result.extract_completions(),
        result.prompt,
        result.metadata,
        model_spec,
        n_samples=n_samples,
        return_logprobs=return_logprobs,
        **kwargs,
    )


def postprocess_sample_freeform(
    completions: list[str],
    prompt: Union[OpenAICreatePrompt, OpenAICreateChatPrompt, Prompt],
    metadata: dict,
    model_spec: ModelSpec,
    *,
    n_samples: Optional[int] = None,
    return_logprobs: bool = False,
    **kwargs,
) -> Union[str, list[str], dict]:
    """
    Records the sampled response, prompt and metedata, and returns the sampled text.
    Typically called after `sample_freeform`.

    ARGS
    ====
    `response`: The result of the API call.
    `prompt`: See `completion_query`.
    `n_samples`: The number of samples to generate (1 if None).
    `return_logprobs`: If True, returns the tokens and corresponding logprobs
        in addition to the sampled text.
    `kwargs`: See `completion_query`.

    RETURNS
    =======
    If `return_logprobs` is True, returns a dict with the sampled text, tokens,
        and corresponding logprobs. If `n_samples` is None, the outer list is
        removed from all values.
    Otherwise, returns the sampled text, or a list of sampled texts if
        `n_samples` is not None.
    """
    if n_samples is None:
        sampled = completions[0]
    record_sampling(prompt=prompt, sampled=sampled, metadata=metadata)

    if return_logprobs:
        assert not model_spec.is_chat, "logprobs only works for non-chat models"
        assert not kwargs.get("logprobs") is None

        def _maybe_tokens(logprobs: Optional[dict]) -> Optional[list[str]]:
            return logprobs["tokens"] if logprobs is not None else None

        def _maybe_logprobs(logprobs: Optional[dict]) -> Optional[list[float]]:
            return logprobs["token_logprobs"] if logprobs is not None else None

        def _maybe_top_logprobs(logprobs: Optional[dict]) -> Optional[list[dict[str, float]]]:
            return [dict(x) for x in logprobs["top_logprobs"]] if logprobs is not None else None

        tokens = [_maybe_tokens(choice["logprobs"]) for choice in response["choices"]]
        logprobs = [_maybe_logprobs(choice["logprobs"]) for choice in response["choices"]]
        top_logprobs = [_maybe_top_logprobs(choice["logprobs"]) for choice in response["choices"]]
        if n_samples is None:
            tokens = tokens[0]
            logprobs = logprobs[0]
            top_logprobs = top_logprobs[0]
        return {
            "text": sampled,
            "tokens": tokens,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
        }

    return sampled
