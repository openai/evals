from corpus import Corpus,NltkCorpus
from collections import namedtuple

Thresholds = namedtuple('Thresholds', ['lower', 'upper'])
LengthBounds = namedtuple('LengthBounds', ['lower', 'upper'])

class CorpusProcessor:

    def __init__(self, corpus: Corpus) -> None:
        self.corpus = corpus

    def parts_of_speech_filter(self, parts_of_speech: list[str]) -> None:
        tagged_words = self.corpus.get_pos_tagged_corpus()
        self.corpus.words = [word for word, pos in tagged_words if pos in parts_of_speech]

    def frequency_filter(self, thresholds: Thresholds = Thresholds(0, float('inf')),
                         filter_corpus: Corpus = None,) -> list[str]:
        if not filter_corpus:
            filter_corpus = NltkCorpus("brown")
        frequency_dist = filter_corpus.get_frequency_distribution()
        lower_bound, upper_bound = thresholds
        self.corpus.words = [word for word in self.corpus if lower_bound <= frequency_dist[word] <= upper_bound]

    def length_filter(self, length_bounds: LengthBounds) -> None:
        lower_bound, upper_bound = length_bounds
        self.corpus.words = [word for word in self.corpus if lower_bound <= len(word) <= upper_bound]

    def subword_filter(self, subword: str) -> None:
        self.corpus.words = [word for word in self.corpus if subword not in word]

