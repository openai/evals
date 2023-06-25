import requests
from abc import ABC, abstractmethod
from typing import Generator, Optional


class RelatedWords(ABC):
    """
    This is an abstract base class for getting related words.

    To implement this class for a specific API, you must define
    a _get_related_words() method.
    """

    def __init__(self, word: str) -> None:
        self.word = word
        self.related_words = self._get_related_words()

    @abstractmethod
    def _get_related_words(self) -> dict[str, str]:
        """
        Abstract method to get related words.

        When implemented, this should return a dictionary mapping words
        to their related words.
        """
        raise NotImplementedError

    def sub_word_filter(self) -> None:
        words_to_remove = [related_word for related_word in self.related_words if self.word in related_word]
        for related_word in words_to_remove:
            del self.related_words[related_word]

    def __repr__(self) -> str:
        return f"DataMuseRelatedWords(word={self.word}, kwargs={self.kwargs})"

    def __len__(self) -> int:
        return len(self.related_words)

    def __getitem__(self, index: int) -> str:
        return self.related_words[index]['word']

    def __contains__(self, item: str) -> bool:
        return any(word_dict["word"] == item for word_dict in self.related_words)

    def __iter__(self) -> Generator[str, None, None]:
        for word_dict in self.related_words:
            yield word_dict['word']


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
            - md: Metadata flags (e.g., d for definitions, p for parts of speech, f for word frequency)

    Returns:
        A dictionary of related words.

    Examples:
        dm = DataMuseRelatedWords("duck", ml="animal", max=5)
        print(dm.related_words)
    """

    def __init__(self, word: str, **kwargs: Optional[str | int]) -> None:
        self.word = word
        self.base_api_url = "https://api.datamuse.com/words"
        self.kwargs = kwargs
        super().__init__(self.word)

    def get_metadata(self, word: str) -> dict[str, str | int | list[str]]:
        for word_dict in self.related_words:
            if word_dict['word'] == word:
                return word_dict
        raise ValueError(f"No metadata found for word: {word}")

    def _get_related_words(self) -> dict[str, str]:
        """
        Sends a request to the DataMuse API and returns the related words.

        Returns:
            A dictionary mapping words to their related words.
        """
        response = requests.get(self.base_api_url, params=self.kwargs)
        return response.json()




if __name__ == "__main__":
    dm = DataMuseRelatedWords("duck", ml="animal", max=5)
    print(dm.related_words)
    print('fauna' in dm)
    print(dm.get_metadata("fauna"))
    for word in dm:
        print(word)
