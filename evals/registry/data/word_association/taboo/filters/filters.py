import nltk
from abc import ABC, abstractmethod


class Filter(ABC):
    @abstractmethod
    def __call__(self, words: list[str]) -> list[str]:
        raise NotImplementedError


class PartsOfSpeechFilter(Filter):
    def __init__(self, parts_of_speech: list[str]) -> None:
        self.parts_of_speech = parts_of_speech

    def __call__(self, words: list[str]) -> list[str]:
        return [word for word in words if word[1] in self.parts_of_speech]


class FrequencyFilter(Filter):
    def __init__(self, frequency_distribution: nltk.FreqDist, threshold: int) -> None:
        self.frequency_distribution = frequency_distribution
        self.threshold = threshold

    def __call__(self, words: list[str]) -> list[str]:
        return [
            word
            for word in words
            if self.frequency_distribution[word] >= self.threshold
        ]


class SubwordFilter(Filter):
    def __init__(self, subword: str) -> None:
        self.subword = subword

    def __call__(self, words: list[str]) -> list[str]:
        return [word for word in words if self.subword not in word]
