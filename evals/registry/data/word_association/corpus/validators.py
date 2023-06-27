from abc import ABC, abstractmethod
from collections.abc import Callable


class QualityValidator(ABC):

    def __init__(self, target_score: str) -> None:
        self.target_score = target_score

    @abstractmethod
    def validate(self, word: str, words: str) -> bool:
        raise NotImplementedError


class EmbeddingsValidator(QualityValidator):

    def validate(self, word: str, words: str, similarity_function: Callable[(str, str), str] = None) -> bool:
        if not similarity_function:
            similarity_function = self.calculate_cosign_similarity
        similarity_function(word, words)
        raise NotImplementedError

    def calculate_cosign_similarity(self, word1: str, word2: str) -> float:
        raise NotImplementedError

    def calculate_euclidean_distance(self, word1: str, word2: str) -> float:
        raise NotImplementedError

