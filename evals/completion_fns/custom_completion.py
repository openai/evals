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

import ssl
import certifi
import json

ssl_context = ssl.create_default_context(cafile=certifi.where())


class CustomCompletionResult(CompletionResult):
    def __init__(self, raw_data: Any, prompt: Any):
        self.raw_data = raw_data
        self.prompt = prompt

    def get_completions(self) -> list[str]:
        completions = []
        if self.raw_data:
            response_text = self.raw_data.get("response", "")
            note_text = self.raw_data.get("note", "")
            full_response = response_text
            if note_text:
                full_response += f"\n{note_text}"

            completions.append(full_response)
        return completions
class CustomCompletionFn(CompletionFnSpec):
    def __init__(
        self,
        api_base: str,
        model: Optional[str] = "gpt-4",
        extra_options: Optional[dict] = {},
        registry: Optional[Any] = None,
    ):
        self.api_base = api_base
        self.model = model
        self.extra_options = extra_options
        self.registry = registry
    def __call__(
        self,
        prompt: Union[str, OpenAICreateChatPrompt],
        **kwargs,
    ) -> CustomCompletionResult:
        if isinstance(prompt, str):
            query_text = prompt
        elif isinstance(prompt, ChatCompletionPrompt):
            query_text = prompt.raw_prompt
        elif isinstance(prompt, list):  # Handle list format
            if all(isinstance(msg, dict) and "content" in msg for msg in prompt):
                query_text = prompt[-1]["content"]  # Extract last message
            else:
                raise ValueError(f"Unsupported prompt format: {type(prompt)}")
        else:
            raise ValueError(f"Unsupported prompt format: {type(prompt)}")
        payload = {"question": query_text}
        logging.info(f"Sending API request with payload: {json.dumps(payload, indent=4)}")
        try:
            response = requests.post(self.api_base, json=payload, proxies={"http": None, "https": None}, timeout=40000, verify=False)
            response.raise_for_status()  # Raises error for non-200 responses
            result_data = response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {e}")
            raise Exception(f"API request failed: {e}")

        result = CustomCompletionResult(raw_data=result_data, prompt=query_text)
        record_sampling(
            prompt=result.prompt,
            sampled=result.get_completions(),
            model=self.model,
            usage=None,
        )

        return result