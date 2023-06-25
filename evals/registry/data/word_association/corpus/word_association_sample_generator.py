import json
from corpus import Corpus, NltkCorpus
from processor import CorpusProcessor, Thresholds, LengthBounds
from related_words import RelatedWords, DataMuseRelatedWords


class EvalTemplate:
    samples = []

    def create_sample(self, system_message: str, user_message: str, ideal_answer: str,) -> dict[str, str | list[str]]:
        sample = {
            "input": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            "ideal": ideal_answer,
        }
        self.samples.append(sample)
        return sample

    def export_to_jsonl(self, filename: str = "samples.jsonl") -> None:
        with open(filename, "w") as f:
            for sample in self.samples:
                f.write(json.dumps(sample) + "\n")


if __name__ == "__main__":
    corpus = NltkCorpus("words")
    freq_filter_corpus = NltkCorpus("brown")
    processor = CorpusProcessor(corpus)
    processor.frequency_filter(thresholds=(50, 10000), filter_corpus=freq_filter_corpus)
    processor.length_filter(length_bounds=(5, 5))
    processor.parts_of_speech_filter(["NN", "VB"])
    corpus = processor.corpus
    for word in corpus:
        # Refactor by creating RelatedWordsProcessor
        related_words = DataMuseRelatedWords(word)
        related_words.sub_word_filter()
        related_words.max_words_filter(1)
        print(word, list(related_words))
