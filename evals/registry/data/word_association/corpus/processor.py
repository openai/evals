from .corpus import Corpus
from collections import namedtuple

Thresholds = namedtuple('Thresholds', ['lower', 'upper'])


class CorpusProcessor:

    def __init__(self, corpus: Corpus) -> None:
        self.corpus = corpus

    def parts_of_speech_filter(self, parts_of_speech: list[str]) -> list[str]:
        tagged_words = self.corpus.get_pos_tagged_corpus()
        self.corpus.words = [word for word, pos in tagged_words if pos in parts_of_speech]

    def frequency_filter(self, thresholds: Thresholds = Thresholds(0, float('inf'))) -> list[str]:
        frequency_dist = self.corpus.get_frequency_distribution()
        lower_bound, upper_bound = thresholds
        self.corpus.words = [word for word in self.corpus if lower_bound <= frequency_dist[word] <= upper_bound]

    def subword_filter(self, subword: str) -> list[str]:
        self.corpus.words = [word for word in self.corpus if subword not in word]

