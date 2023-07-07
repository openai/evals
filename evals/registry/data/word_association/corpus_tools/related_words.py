"""
This module provides classes to find related words for a given word using various APIs. The main class is RelatedWords,
which is an abstract base class and should not be used directly. Instead, users should use derived classes that
implement the functionality for specific APIs. Currently, this module supports the DataMuse API.

Classes:
    RelatedWords: Abstract base class for getting related words.
    DataMuseRelatedWords: A class to get related words using the DataMuse API.
    GPTRelatedWords: A class to get related words using ChatGPT Completions API. (Not yet implemented)
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import requests


class RelatedWords(ABC):
    """An abstract base class for getting related words.

    To implement this class for a specific API, you must define a _get_related_words() method.
    """

    def __init__(self, word: str, **kwargs: Optional[Union[str, int]]) -> None:
        self.word = word
        self.kwargs = kwargs
        self.words_dict = self._get_related_words()
        self.words = [word["word"] for word in self.words_dict]

    @abstractmethod
    def _get_related_words(self) -> List[Dict[str, Any]]:
        """
        Abstract method to get related words.

        When implemented, this should return a list of dictionaries mapping words
        to their related words.
        """
        raise NotImplementedError

    def get_pos_tagged_words(self) -> List[Tuple[str, str]]:
        """
        Method to get part-of-speech tagged words.

        The method should be implemented in a derived class.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        """
        Return the string representation of a RelatedWords instance.
            type(self)(word={self.word}, kwargs={self.kwargs})
        """
        return f"{type(self)}(word={self.word}, kwargs={self.kwargs})"

    def __len__(self) -> int:
        """
        Get the length of the related words collection.

        Returns:
            The number of related words.
        """
        return len(self.words)

    def __getitem__(self, index: int) -> str:
        """
        Get the word at the specified index in the related word collection.
        """
        return self.words[index]

    def __contains__(self, item: str) -> bool:
        """
        Check if a word is in the related word collection.
        """
        return item in self.words

    def __iter__(self) -> Generator[str, None, None]:
        """
        Create an iterator for the related word collection.
        """
        for word in self.words:
            yield word


class DataMuseRelatedWords(RelatedWords):
    """
    A class to get related words using the DataMuse API.
    Docs: https://www.datamuse.com/api/
    Endpoint: https://api.datamuse.com/words
    You can use this service without restriction and without an API key for up to 100,000 requests per day

    Args:
        word (str): The word to find related words for.
        constraint (str): Constraint for related words, rel_[code] Default: 'rel_syn'.
            [code] | Description | Example
            jja | Popular nouns modified by the given adjective, per Google Books Ngrams | gradual → increase
            jjb | Popular adjectives used to modify the given noun, per Google Books Ngrams | beach → sandy
            syn | Synonyms (words contained within the same WordNet synset) | ocean → sea
            trg | "Triggers" (words that are statistically associated with the query word) | cow → milking
            ant | Antonyms (per WordNet) | late → early
            spc | "Kind of" (direct hypernyms, per WordNet) | gondola → boat
            gen | "More general than" (direct hyponyms, per WordNet) | boat → gondola
            com | "Comprises" (direct holonyms, per WordNet) | car → accelerator
            par | "Part of" (direct meronyms, per WordNet) | trunk → tree
            bga | Frequent followers (w′ such that P(w′|w) ≥ 0.001, per Google Books Ngrams) | wreak → havoc
            bgb | Frequent predecessors (w′ such that P(w|w′) ≥ 0.001, per Google Books Ngrams) | havoc → wreak
            rhy | Rhymes ("perfect" rhymes, per RhymeZone) | spade → aid
            nry | Approximate rhymes (per RhymeZone) | forest → chorus
            hom | Homophones (sound-alike words) | course → coarse
            cns | Consonant match | sample → simple

        **kwargs: Additional parameters for the API. Valid parameters are:
            - ml: Means like constraint
            - sl: Sounds like constraint
            - sp: Spelled like constraint
            - rel_[code]: Related word constraints (e.g., rel_jjb for adjectives
                          that modify the given noun)
            - v: Identifier for the vocabulary to use
            - topics: Topic words to skew results towards
            - lc: The word that appears to the left of the target word
            - rc: The word that appears to the right of the target word
            - max: Maximum number of results to return
            - md: Metadata flags # Warning if you override this kwarg and don't include
                (p for parts of speech or f for word frequency, calls to those filters will fail)
    """

    def __init__(
        self,
        word: str,
        constraint: str = "rel_syn",
        **kwargs: Optional[Union[str, int]],
    ) -> None:
        self.constraint = constraint
        super().__init__(word, **kwargs)

    def get_pos_tagged_words(self) -> List[Tuple[str, str]]:
        """
        Get the part-of-speech tagged words from the related words collection.

        Returns:
            A list of tuples, where each tuple contains a word and its part-of-speech tag.
        """
        tagged_words = []
        for word in self.words:
            word_meta_data = self.get_metadata(word)
            tag_pair = word, word_meta_data["tags"][0]
            tagged_words.append(tag_pair)
        return tagged_words

    def get_metadata(self, word: str) -> Dict[str, Union[str, int, List[str]]]:
        """
        Get the metadata associated with a word in the related words' collection.

        Args:
            word (str): The word for which metadata is to be retrieved.

        Returns:
            A dictionary containing the metadata for the specified word.

        Raises:
            ValueError: If no metadata is found for the specified word.
        """
        for word_dict in self.words_dict:
            if word_dict["word"] == word:
                return word_dict
        raise ValueError(f"No metadata found for word: {word}")

    def _get_related_words(self) -> List[Dict[str, str]]:
        """
        Sends a request to the DataMuse API and returns the related words.

        Returns:
            A list of dictionaries containing related words and metadata.
        """
        params = {"md": "frspd"}
        params.update(self.kwargs)
        response = requests.get(
            f"https://api.datamuse.com/words?{self.constraint}={self.word}",
            params=params,
        )
        return response.json()


class GPTGeneratedRelatedWords(RelatedWords):
    """A class to get related words using ChatGPT Completions API. (Not yet implemented)"""

    def _get_related_words(self) -> List[Dict[str, Any]]:
        raise NotImplementedError


if __name__ == "__main__":
    dm = DataMuseRelatedWords("duck", max=5)
    print(dm.words)
    print("fauna" in dm)
    print(dm.get_metadata("fauna"))
    for word in dm:
        print(word)
