"""
This file provides common interfaces and utilities used by eval creators to
sample from models and process the results.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Protocol, Union, runtime_checkable

from evals.prompt.base import OpenAICreateChatPrompt, OpenAICreatePrompt, Prompt
from evals.record import record_match

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
            the appropriate `Prompt` class.
        `kwargs`: Other arguments passed to the API.

        RETURNS
        =======
        The result of the API call.
        The prompt that was fed into the API call as a str.
        """


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
