import nltk
from .corpus import Corpus
from .filters import parts_of_speech_filter, frequency_filter, subword_filter


class CorpusPipeline:
    def __init__(self, corpus: Corpus) -> None:
        self.corpus = corpus

    def tag_parts_of_speech(self, func: callable = None) -> list[tuple[str, str]]:
        if not func:
            nltk.download("averaged_perceptron_tagger")
            func = nltk.pos_tag
        return func(self.corpus)

    def parts_of_speech_filter(self, parts_of_speech: list[str] = None) -> list[str]:
        if not parts_of_speech:
            parts_of_speech = ["NN", "VB"]
        tagged_corpus = self.tag_parts_of_speech()
        word_list = [word for word, pos in tagged_corpus if pos in parts_of_speech]
        return word_list
        # NOTE: need to change this process to use a filter class

    def frequency_filter(self) -> list[str]:
        raise NotImplementedError

    def subword_filter(self) -> list[str]:
        raise NotImplementedError

    def run(self) -> Corpus:
        self.tag_parts_of_speech()
        self.parts_of_speech_filter()
        self.frequency_filter()
        self.subword_filter()
        return self.corpus
