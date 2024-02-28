import os
from typing import Any, Optional, Union

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
from evals.utils.api_utils import (
    openai_chat_completion_create_retrying,
    openai_completion_create_retrying,
    openai_rag_completion_create_retrying
)
from evals.completion_fns.openai import OpenAICompletionResult, OpenAIChatCompletionResult
from evals.utils.doc_utils import extract_text_and_fill_in_images, extract_text, num_tokens_from_string


model_max_tokens = {
    "gpt-3.5-turbo": 15800,
}


class OpenAIBaseCompletionResult(CompletionResult):
    def __init__(self, raw_data: Any, prompt: Any):
        self.raw_data = raw_data
        self.prompt = prompt

    def get_completions(self) -> list[str]:
        raise NotImplementedError


class OneAPICompletionFn(CompletionFn):
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
        self.api_base = api_base or os.environ.get("ONEAPI_BASE", None)
        self.api_key = api_key or os.environ.get("ONEAPI_KEY", None)
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

        if "file_name" in kwargs:
            attached_file_content = "\nThe file is as follows:\n\n" + "".join(extract_text(kwargs["file_name"]))
        else:
            attached_file_content = ""
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith("file")}

        openai_create_prompt += attached_file_content

        result = openai_completion_create_retrying(
            OpenAI(api_key=self.api_key, base_url=self.api_base),
            model=self.model,
            prompt=openai_create_prompt,
            **{**kwargs, **self.extra_options},
        )
        result = OpenAICompletionResult(raw_data=result, prompt=openai_create_prompt)
        record_sampling(prompt=result.prompt, sampled=result.get_completions())
        return result


class OneAPIChatCompletionFn(CompletionFnSpec):
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
        self.api_base = api_base or os.environ.get("ONEAPI_BASE", None)
        self.api_key = api_key or os.environ.get("ONEAPI_KEY", None)
        self.n_ctx = n_ctx
        self.extra_options = extra_options
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)

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

        if "file_name" in kwargs:
            attached_file_content = "\nThe file is as follows:\n\n" + "".join(extract_text(kwargs["file_name"]))
            max_tokens = model_max_tokens.get(self.model, 1000000)
            while num_tokens_from_string(attached_file_content, "cl100k_base") > max_tokens:
                attached_file_content = attached_file_content[:-1000]
        else:
            attached_file_content = ""
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith("file")}

        openai_create_prompt[-1]["content"] += attached_file_content
        # for message in openai_create_prompt:
        #     message["role"] = "human" if message["role"] == "user" else message["role"]

        result = openai_chat_completion_create_retrying(
            self.client,
            model=self.model,
            messages=openai_create_prompt,
            **{**kwargs, **self.extra_options},
        )
        result = OpenAIChatCompletionResult(raw_data=result, prompt=openai_create_prompt)
        record_sampling(prompt=result.prompt, sampled=result.get_completions())
        print(result.get_completions())
        return result
