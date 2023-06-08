from typing import Union
from evals.api import CompletionFn, CompletionResult
from evals.prompt.base import OpenAICreateChatPrompt, OpenAICreatePrompt, Prompt

class TestCompletionResult(CompletionResult):
    def __init__(self, completion: str):
        self.completion = completion

    def get_completions(self) -> list[str]:
        return [self.completion]


class TestCompletionFn(CompletionFn):
    def __init__(self, completion: str):
        self.completion = completion

    def __call__(
        self, prompt: Union[OpenAICreatePrompt, OpenAICreateChatPrompt, Prompt], **kwargs
    ) -> CompletionResult:
        return TestCompletionResult(self.completion)