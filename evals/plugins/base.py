from abc import ABC, abstractmethod

import pydantic

from evals.base import ModelSpec
from evals.prompt.base import OpenAICreateChatPrompt, OpenAICreatePrompt


class _ModelRunner(pydantic.BaseModel, ABC):
    @classmethod
    @abstractmethod
    def resolve(cls, name: str) -> ModelSpec:
        raise NotImplementedError

    def completion(self, prompt: OpenAICreatePrompt, **kwargs):
        raise NotImplementedError

    def chat_completion(self, message: OpenAICreateChatPrompt, **kwargs):
        raise NotImplementedError
