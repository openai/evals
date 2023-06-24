import nltk
from abc import ABC, abstractmethod


class Corpus(ABC):
    def __init__(self, name: str) -> None:
        self.name = name
        self.words = self._get_corpus()

    @abstractmethod
    def _get_corpus(self) -> list[str]:
        raise NotImplementedError

    def apply_filters(self, filters: list[callable]) -> list[str]:
        for filtr in filters:
            self.words = filtr(self.words)

    def __len__(self) -> int:
        return len(self.words)

    def __getitem__(self, index: int) -> str:
        return self.words[index]

    def __setitem__(self, index: int, value: str) -> None:
        self.words[index] = value

    def __delitem__(self, index: int) -> str:
        del self.words[index]

    def __iter__(self) -> iter:
        return iter(self.words)

    def __contains__(self, word: str) -> bool:
        return word in self.words

    def __repr__(self) -> str:
        return f"Corpus({self.name})"


class NltkCorpus(Corpus):
    def __init__(self, nltk_corpus: str) -> None:
        self.nltk_corpus = nltk_corpus
        super().__init__(name=self.nltk_corpus)

    def _get_corpus(self) -> list[str]:
        nltk.download(self.nltk_corpus)
        corpus = getattr(nltk.corpus, self.nltk_corpus)
        return corpus.words()

    def get_frequency_distribution(self) -> nltk.FreqDist:
        return nltk.FreqDist(self.words)

    def get_pos_tagged_corpus(self) -> list[tuple[str, str]]:
        nltk.download("averaged_perceptron_tagger")
        return nltk.pos_tag(self.words)





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

