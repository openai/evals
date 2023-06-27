import openai
import os
import numpy as np
from abc import ABC, abstractmethod
from collections.abc import Callable
from evals.completion_fns.openai import OpenAIChatCompletionFn

openai.api_key = os.environ.get("OPENAI_API_KEY")


class QualityValidator(ABC):

    def __init__(self, target_score: int) -> None:
        self.target_score = target_score

    @abstractmethod
    def validate(self, word: str, words: str) -> bool:
        raise NotImplementedError


class EmbeddingsValidator(QualityValidator):
    # Refactor this to take in word, related word dictionary pairs

    def validate(self, word: str, words: str, similarity_function: Callable[(str, str), str] = None) -> bool:
        if not similarity_function:
            similarity_function = self.calculate_cosine_similarity
        word_vector = self.get_embedding(word)
        related_words_vectors = self.get_embedding(words)
        value: int = similarity_function(word_vector, related_words_vectors)
        return value >= self.target_score

    def calculate_cosine_similarity(self, vec1: str, vec2: str) -> float:
        vec1_norm = np.linalg.norm(vec1)
        vec2_norm = np.linalg.norm(vec2)
        similarity = np.dot(vec1, vec2) / (vec1_norm * vec2_norm)

        return similarity

    def calculate_euclidean_distance(self, word1: str, word2: str) -> float:
        raise NotImplementedError

    def get_embedding(self, word: str) -> list[float]:
        # Should batch this
        response = openai.Embedding.create(model="text-embedding-ada-002", input=word)
        return response["data"][0]["embedding"]


class GPTValidator(QualityValidator):

    def validate(self, word: str, words: str) -> bool:
        raise NotImplementedError

    def get_chat_completion(self, word: str, words: str) -> bool:
        raise NotImplementedError


if __name__ == "__main__":
    validator = EmbeddingsValidator(0.75)
    embedding = validator.get_embedding("world")
    embedding2 = validator.get_embedding("globe, earth, man, planet")
    similarity = validator.calculate_cosine_similarity(embedding, embedding2)
    print(similarity)
    validated = validator.validate("world", "globe, earth, man, planet")
    print(validated)
