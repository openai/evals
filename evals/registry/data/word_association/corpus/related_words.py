import requests
import nltk
from abc import ABC, abstractmethod
from typing import Generator, Optional
from collections.abc import Callable


class RelatedWords(ABC):
    """
    This is an abstract base class for getting related words.

    To implement this class for a specific API, you must define
    a _get_related_words() method.
    """

    def __init__(self, word: str, **kwargs: Optional[str | int]) -> None:
        self.word = word
        self.kwargs = kwargs
        self.words = self._get_related_words()
        self.words_dict = self.words.copy()
        self.words = [word["word"] for word in self.words]

    @abstractmethod
    def _get_related_words(self) -> dict[str, str]:
        """
        Abstract method to get related words.

        When implemented, this should return a dictionary mapping words
        to their related words.
        """
        raise NotImplementedError

    def get_pos_tagged_words(self) -> list[tuple[str, str]]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self)}(word={self.word}, kwargs={self.kwargs})"

    def __len__(self) -> int:
        return len(self.words)

    def __getitem__(self, index: int) -> str:
        return self.words[index]

    def __contains__(self, item: str) -> bool:
        return item in self.words

    def __iter__(self) -> Generator[str, None, None]:
        for word in self.words:
            yield word


class DataMuseRelatedWords(RelatedWords):
    """
    This class gets related words using the DataMuse API.
    Docs: https://www.datamuse.com/api/
    Endpoint: https://api.datamuse.com/words
    You can use this service without restriction and without an API key for up to 100,000 requests per day

    Args:
        word (str): The word to find related words for.
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

    Returns:
        A dictionary of related words.

    Examples:
        dm = DataMuseRelatedWords("duck", ml="animal", max=5)
        print(dm.related_words)
    """

    def __init__(self, word: str, constraint: str = "rel_syn", **kwargs: Optional[str | int]) -> None:
        self.constraint = constraint
        super().__init__(word, **kwargs)

    def get_pos_tagged_words(self) -> list[tuple[str, str]]:
        tagged_words = []
        for word in self.words:
            word_meta_data = self.get_metadata(word)
            tag_pair = word, word_meta_data["tags"][0]
            tagged_words.append(tag_pair)
        return tagged_words

    def get_metadata(self, word: str) -> dict[str, str | int | list[str]]:
        for word_dict in self.words_dict:
            if word_dict['word'] == word:
                return word_dict
        raise ValueError(f"No metadata found for word: {word}")

    def _get_related_words(self) -> dict[str, str]:
        """
        Sends a request to the DataMuse API and returns the related words.

        Returns:
            A dictionary mapping words to their related words.
        """
        params = {"md": "frspd"}
        params.update(self.kwargs)
        response = requests.get(f"https://api.datamuse.com/words?{self.constraint}={self.word}", params=params)
        return response.json()


if __name__ == "__main__":
    dm = DataMuseRelatedWords("duck", max=5)
    print(dm.words)
    print('fauna' in dm)
    print(dm.get_metadata("fauna"))
    for word in dm:
        print(word)
