from evals.registry.data.word_association.corpus.corpus import Corpus, NltkCorpus
from evals.registry.data.word_association.corpus.processor import CorpusProcessor, Thresholds
from evals.registry.data.word_association.corpus.related_words import RelatedWords, DataMuseRelatedWords


class CorpusEvalSampleGenerator:
    def __init__(self, corpus: Corpus) -> None:
        self.corpus = corpus
        self.corpus_processor = CorpusProcessor(self.corpus)



class WordAssociationEvalSampleGenerator:
    pass


class TabooEvalSampleGenerator:
    pass


