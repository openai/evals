"""
Extending Completion Functions with Embeddings-based retrieval from a fetched dataset
"""
from ast import literal_eval
from typing import Any, Optional, Union

import numpy as np
import openai
import pandas as pd

from evals.api import CompletionFn, CompletionResult
from evals.prompt.base import ChatCompletionPrompt, CompletionPrompt
from evals.record import record_sampling
from evals.registry import Registry


def load_embeddings(embeddings_and_text_path: str):
    df = pd.read_csv(embeddings_and_text_path, converters={"embedding": literal_eval})
    assert (
        "text" in df.columns and "embedding" in df.columns
    ), "The embeddings file must have columns named 'text' and 'embedding'"
    return df


def find_top_k_closest_embeddings(embedded_prompt: list[float], embs: list[list[float]], k: int):
    # Normalize the embeddings
    norm_embedded_prompt = embedded_prompt / np.linalg.norm(embedded_prompt)
    norm_embs = embs / np.linalg.norm(embs, axis=1)[:, np.newaxis]

    # Calculate cosine similarity
    cosine_similarities = np.dot(norm_embs, norm_embedded_prompt)

    # Get the indices of the top k closest embeddings
    top_k_indices = np.argsort(cosine_similarities)[-k:]

    return top_k_indices[::-1]


DEFAULT_RETRIEVAL_TEMPLATE = "Use the provided context to answer the question. "


class RetrievalCompletionResult(CompletionResult):
    def __init__(self, response: str) -> None:
        self.response = response

    def get_completions(self) -> list[str]:
        return [self.response.strip()]


class RetrievalCompletionFn(CompletionFn):
    """
    This Completion Function uses embeddings to retrieve the top k relevant docs from a dataset to the prompt, then adds them to the context before calling the completion.
    """

    def __init__(
        self,
        completion_fn: str,
        embeddings_and_text_path: str,
        retrieval_template: str = DEFAULT_RETRIEVAL_TEMPLATE,
        k: int = 4,
        embedding_model: str = "text-embedding-ada-002",
        registry: Optional[Registry] = None,
        registry_path: Optional[str] = None,
        **_kwargs: Any
    ) -> None:
        """
        Args:
            retrieval_template: The template to use for the retrieval. The task prompt will be added to the end of this template.
            k: The number of docs to retrieve from the dataset.
            completion_fn: The completion function to use for the retrieval.
            embeddings_and_text_path: The path to a CSV containing "text" and "embedding" columns.
            registry: Upstream callers may pass in a registry to use.
            registry_path: The path to a registry file to add to default registry.
            _kwargs: Additional arguments to pass to the completion function instantiation.
        """
        registry = Registry() if not registry else registry
        if registry_path:
            registry.add_registry_paths(registry_path)

        self.embeddings_df = load_embeddings(embeddings_and_text_path)

        self.embedding_model = embedding_model
        self.k = k

        self.retrieval_template = retrieval_template
        self.completion_fn_instance = registry.make_completion_fn(completion_fn)

    def __call__(self, prompt: Union[str, list[dict]], **kwargs: Any) -> RetrievalCompletionResult:
        """
        Args:
            prompt: The prompt to complete, in either text string or Chat format.
            kwargs: Additional arguments to pass to the completion function call method.
        """
        # Embed the prompt
        embedded_prompt = openai.Embedding.create(
            model=self.embedding_model, input=CompletionPrompt(prompt).to_formatted_prompt()
        )["data"][0]["embedding"]

        embs = self.embeddings_df["embedding"].to_list()

        # Compute the cosine similarity between the prompt and the embeddings
        topk = " ".join(
            self.embeddings_df.iloc[
                find_top_k_closest_embeddings(embedded_prompt, embs, k=self.k)
            ].text.values
        )

        prompt = ChatCompletionPrompt(prompt).to_formatted_prompt()
        retrieval_prompt = [{"role": "system", "content": self.retrieval_template + topk}] + prompt

        answer = self.completion_fn_instance(prompt=retrieval_prompt, **kwargs).get_completions()[0]
        record_sampling(prompt=retrieval_prompt, sampled=answer)
        return RetrievalCompletionResult(answer)
