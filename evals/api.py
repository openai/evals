"""
This file provides common interfaces and utilities used by eval creators to
sample from models and process the results.
"""

import logging
from typing import Callable, Optional, Union

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


def completion_query(
    model_spec: ModelSpec,
    prompt: Union[OpenAICreatePrompt, OpenAICreateChatPrompt, Prompt],
    **kwargs,
) -> tuple[dict, Union[OpenAICreatePrompt, OpenAICreateChatPrompt], dict]:
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

        if model_spec.is_chat:
            for choice in result["choices"]:
                choice["text"] = choice["message"]["content"]

    return result, openai_create_prompt, metadata


def check_sampled_text(
    model_spec: ModelSpec,
    prompt: Union[OpenAICreatePrompt, OpenAICreateChatPrompt, Prompt],
    expected: Union[str, list[str], tuple[str]],
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
    if isinstance(expected, tuple):
        expected = list(expected)
    elif not isinstance(expected, list):
        expected = [expected]
    if options is None:
        options = expected

    result, actual_prompt, metadata = completion_query(
        prompt=prompt,
        temperature=0.0,
        model_spec=model_spec,
    )
    choice = result["choices"][0]

    sampled = choice["text"].strip() if model_spec.strip_completion else choice["text"]

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
        "prompt": actual_prompt,
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
    temperature: float = 1.0,
    top_p: float = 0.9,
    max_tokens: int = 512,
    stop: Optional[str] = None,
    n_samples: int = None,
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
    response, actual_prompt, metadata = completion_query(
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
    sampled = [choice["text"] for choice in response["choices"]]
    if n_samples is None:
        sampled = sampled[0]
    record_sampling(prompt=actual_prompt, sampled=sampled, metadata=metadata)

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
