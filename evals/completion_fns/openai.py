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


class RetrievalCompletionResult(CompletionResult):
    def __init__(self, response: str, prompt: Any) -> None:
        self.response = response
        self.prompt = prompt

    def get_completions(self) -> list[str]:
        return [self.response.strip()]


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

        if "file_name" not in kwargs:
            result = openai_completion_create_retrying(
                OpenAI(api_key=self.api_key, base_url=self.api_base),
                model=self.model,
                prompt=openai_create_prompt,
                **{**kwargs, **self.extra_options},
            )
            result = OpenAICompletionResult(raw_data=result, prompt=openai_create_prompt)
        else:
            answer = openai_rag_completion_create_retrying(
                OpenAI(api_key=self.api_key, base_url=self.api_base),
                model=self.model,
                instructions=kwargs.get("instructions", ""),
                file_name=kwargs.get("file_name", ""),
            )
            result = RetrievalCompletionResult(answer, prompt=openai_create_prompt)
        record_sampling(prompt=result.prompt, sampled=result.get_completions())
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

        if "file_name" not in kwargs:
            result = openai_chat_completion_create_retrying(
                OpenAI(api_key=self.api_key, base_url=self.api_base),
                model=self.model,
                messages=openai_create_prompt,
                **{**kwargs, **self.extra_options},
            )
            result = OpenAIChatCompletionResult(raw_data=result, prompt=openai_create_prompt)
        else:
            chatmodel_to_apimodel = lambda x: "gpt-3.5-turbo-1106" if x.startswith("gpt-3.5-turbo") else "gpt-4-1106-preview" if x.startswith("gpt-4") else ""
            answer = openai_rag_completion_create_retrying(
                OpenAI(api_key=self.api_key, base_url=self.api_base),
                model=chatmodel_to_apimodel(self.model),
                instructions=kwargs.get("instructions", ""),
                file_name=kwargs.get("file_name", ""),
                prompt=CompletionPrompt(raw_prompt=openai_create_prompt).to_formatted_prompt()
            )
            result = RetrievalCompletionResult(answer, prompt=openai_create_prompt)
        record_sampling(prompt=result.prompt, sampled=result.get_completions())
        return result
