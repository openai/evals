"""
This module provides a WordCollectionProcessor class to process and filter collections of words from
either a corpus or a list of related words. The class offers various filtering methods for refining
the word collection based on given criteria such as parts of speech, frequency, character length,
subword presence, and maximum word count.

Classes:
    - WordCollectionProcessor: Processes and filters collections of words based on specified criteria.
"""
from collections import namedtuple
from typing import Iterator, List, Union

from corpus import Corpus, NltkCorpus
from related_words import RelatedWords

Thresholds = namedtuple("Thresholds", ["lower", "upper"])
LengthBounds = namedtuple("LengthBounds", ["lower", "upper"])


class WordCollectionProcessor:
    """
    A class to process a collection of words from either a corpus or a list of related words.
    This class provides various filtering methods to narrow down the collection based on specific criteria.

    Args:
        words (Corpus | RelatedWords): The corpus or list of related words to be processed.
    """

    def __init__(self, words: Union[Corpus, RelatedWords]) -> None:
        self.words = words

    def parts_of_speech_filter(self, parts_of_speech: List[str]) -> None:
        """
        Filters words in the collection by specified parts of speech.
        Refactors to nltk tagging if NotImplemented.

        Args:
            parts_of_speech (list[str]): A list of allowed parts of speech (e.g., "NN", "VBG").
        """
        # Refactor to have default to nltk tagging if NotImplemented
        tagged_words = self.words.get_pos_tagged_words()
        self.words.words = [word for word, pos in tagged_words if pos in parts_of_speech]

    def frequency_filter(
        self, thresholds: Thresholds = Thresholds(0, float("inf")), filter_corpus: Corpus = None
    ) -> None:
        """
        Filters words in the collection by their frequency within the specified thresholds.

        Args:
            thresholds (Thresholds): A tuple of lower and upper bounds for the word frequency filter.
            filter_corpus (Corpus): The corpus to use as the basis for frequency calculation.
                                    Default: Brown corpus from NLTK.
        """
        if filter_corpus is None:
            filter_corpus = NltkCorpus("brown")
        frequency_dist = filter_corpus.get_frequency_distribution()
        lower_bound, upper_bound = thresholds
        self.words.words = [
            word for word in self.words if lower_bound <= frequency_dist[word] <= upper_bound
        ]

    def char_length_filter(self, length_bounds: LengthBounds) -> None:
        """
        Filters words in the collection by their character length within the specified bounds.

        Args:
            length_bounds (LengthBounds): A tuple of lower and upper bounds for the word length filter.
        """
        lower_bound, upper_bound = length_bounds
        self.words.words = [word for word in self.words if lower_bound <= len(word) <= upper_bound]

    def sub_word_filter(self, subword: str) -> None:
        """
        Filters words in the collection by excluding those that contain the given subword.

        Args:
            subword (str): The substring to exclude from the words in the collection.
        """
        self.words.words = [word for word in self.words if subword not in word]

    def str_max_word_count_filter(self, max_num_words: int = 1) -> None:
        """
        Filters words in the collection based on the maximum number of words allowed per element.

        Args:
            max_num_words (int): The maximum number of words allowed per element. Default: 1.
        """
        words_to_remove = [word for word in self.words if word.count(" ") > max_num_words - 1]
        self.words.words = [word for word in self.words if word not in words_to_remove]

    def __iter__(self) -> Iterator[str]:
        """Retrieve an iterator over the words."""
        return iter(self.words)

    def __len__(self) -> int:
        """Retrieve the number of words in the collection."""
        return len(self.words)

    def __getitem__(self, index: int) -> str:
        """Get word from corpus at given index."""
        return self.words[index]
