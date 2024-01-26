"""
Extending Completion Functions with Embeddings-based retrieval from a fetched dataset
"""
import os
from ast import literal_eval
import time
from typing import Any, Optional, Union

import numpy as np
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

from evals.api import CompletionFn, CompletionResult
from evals.completion_fns.openai import RetrievalCompletionResult
from evals.prompt.base import ChatCompletionPrompt, CompletionPrompt
from evals.record import record_sampling
from evals.utils.api_utils import openai_rag_completion_create_retrying


class OpenAIRetrievalCompletionFn(CompletionFn):
    """
    This Completion Function uses embeddings to retrieve the top k relevant docs from a dataset to the prompt, then adds them to the context before calling the completion.
    """

    def __init__(
            self,
            model: Optional[str] = None,
            instructions: Optional[str] = "You are a helpful assistant on extracting information from files.",
            api_base: Optional[str] = None,
            api_key: Optional[str] = None,
            n_ctx: Optional[int] = None,
            extra_options: Optional[dict] = {},
            **kwargs
    ):
        self.model = model
        self.instructions = instructions
        self.api_base = api_base
        self.api_key = api_key
        self.n_ctx = n_ctx
        self.extra_options = extra_options

    def __call__(self, prompt: Union[str, list[dict]], **kwargs: Any) -> RetrievalCompletionResult:
        """
        Args:
            prompt: The prompt to complete, in either text string or Chat format.
            kwargs: Additional arguments to pass to the completion function call method.
        """

        assert "file_name" in kwargs, "Must provide a file_name to retrieve."

        answer = openai_rag_completion_create_retrying(
            client,
            model=self.model,
            instructions=self.instructions,
            file_name=kwargs.get("file_name", ""),
            prompt=CompletionPrompt(raw_prompt=prompt).to_formatted_prompt(),
        )
        record_sampling(prompt=prompt, sampled=answer)
        return RetrievalCompletionResult(answer, prompt=prompt)
