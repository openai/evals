"""
This module provides an abstract base class `Corpus` for working with corpora
and a concrete implementation `NltkCorpus` that uses NLTK to download and
work with NLTK-supported corpora.
"""
from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Tuple

import nltk


class Corpus(ABC):
    """
    An abstract base class representing a corpus of words.
    Define the method _get_corpus in any derived class.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.words = self._get_corpus()

    @abstractmethod
    def _get_corpus(self) -> List[str]:
        """Get the words of the corpus."""
        raise NotImplementedError

    def get_frequency_distribution(self) -> Dict[str, int]:
        """Get a frequency distribution of words."""
        raise NotImplementedError

    def get_pos_tagged_words(self) -> List[Tuple[str, str]]:
        """Get a list of part-of-speech tagged words."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Get corpus length."""
        return len(self.words)

    def __getitem__(self, index: int) -> str:
        """Get word from corpus at given index."""
        return self.words[index]

    def __setitem__(self, index: int, value: str) -> None:
        """Set the value of the word at given index."""
        self.words[index] = value

    def __delitem__(self, index: int) -> None:
        """Delete the word at given index."""
        del self.words[index]

    def __iter__(self) -> Iterator[str]:
        """Retrieve an iterator over the words."""
        return iter(self.words)

    def __contains__(self, word: str) -> bool:
        """Check if the word is present in the corpus."""
        return word in self.words

    def __repr__(self) -> str:
        """Return string representation of the corpus."""
        return f"Corpus({self.name})"


class NltkCorpus(Corpus):
    """
    A concrete implementation of the Corpus class using the NLTK library.
    Downloads and works with NLTK-supported corpora.

    Args:
        nltk_corpus (str): The name of the NLTK corpus to download and use.
    """

    def __init__(self, nltk_corpus: str) -> None:
        self.nltk_corpus = nltk_corpus
        nltk.download(self.nltk_corpus)
        nltk.download("averaged_perceptron_tagger")
        super().__init__(name=self.nltk_corpus)
        self.freq_dist_cache = None
        self.pos_tagged_words_cache = None

    def _get_corpus(self) -> List[str]:
        """Get corpus from NLTK."""
        corpus = getattr(nltk.corpus, self.nltk_corpus)
        return corpus.words()

    def get_frequency_distribution(self) -> nltk.FreqDist:
        """Get a frequency distribution of words using NLTK."""
        if self.freq_dist_cache is None:
            self.freq_dist_cache = nltk.FreqDist(self.words)
        return self.freq_dist_cache

    def get_pos_tagged_words(self) -> List[Tuple[str, str]]:
        """Get a list of part-of-speech tagged words using NLTK."""
        if self.pos_tagged_words_cache is None:
            self.pos_tagged_words_cache = nltk.pos_tag(self.words)
        return self.pos_tagged_words_cache


if __name__ == "__main__":
    corpus = NltkCorpus("words")
    print(corpus)
    for word in corpus:
        print(word)
    print(len(corpus))
    print("hello" in corpus)
    print("hello" not in corpus)
    print(corpus[0])
    del corpus[0]
    print(len(corpus))
