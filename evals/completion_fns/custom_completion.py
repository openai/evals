import logging
from typing import Any, Optional, Union
import requests
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

class CustomCompletionResult(CompletionResult):
    def __init__(self, raw_data: Any, prompt: Any):
        self.raw_data = raw_data
        self.prompt = prompt

    def get_completions(self) -> list[str]:
        completions = []
        if self.raw_data:
            completions.append(self.raw_data.get("response", ""))
        return completions

class CustomCompletionFn(CompletionFnSpec):
    def __init__(
        self,
        api_base: str,
        api_key: Optional[str] = None,
        model: Optional[str] = "gpt-4",
        extra_options: Optional[dict] = {},
        registry: Optional[Any] = None,
    ):
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.extra_options = extra_options
        self.registry = registry
    def __call__(
        self,
        prompt: Union[str, OpenAICreateChatPrompt],
        **kwargs,
    ) -> CustomCompletionResult:
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

        payload = {
            "prompt": openai_create_prompt,
            "model": self.model,
            **self.extra_options,
        }

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        response = requests.post(self.api_base, json=payload, headers=headers)

        if response.status_code != 200:
            logging.warning(f"API request failed with status code {response.status_code}: {response.text}")
            raise Exception(f"API request failed: {response.text}")

        result = response.json()
        result = CustomCompletionResult(raw_data=result, prompt=openai_create_prompt)

        record_sampling(
            prompt=result.prompt,
            sampled=result.get_completions(),
            model=self.model,
            usage=None,
        )

        return result