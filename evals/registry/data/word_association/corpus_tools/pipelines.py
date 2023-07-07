from collections.abc import Callable

from corpus import Corpus


class CorpusPipeline:
    def __init__(self, corpus: Corpus) -> None:
        self.corpus = corpus
        self.operations = []

    def add_operation(self, operation: Callable[Corpus, ...]) -> "CorpusPipeline":
        self.operations.append(operation)
        # for method chaining
        return self

    def run(self) -> Corpus:
        result = self.corpus
        for operation in self.operations:
            result = operation(result)
        return result
