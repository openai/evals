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
    processor.char_length_filter(length_bounds=(5, 5))
    processor.parts_of_speech_filter(["NN", "VB"])

    corpus = processor.corpus
    for word in corpus:
        related_words = DataMuseRelatedWords(word)
        related_processor = CorpusProcessor(related_words)
        # related_processor.char_length_filter(length_bounds=(5, 5))
        related_processor.parts_of_speech_filter(["n", "v"])
        related_processor.str_max_word_count_filter(1)
        related_processor.sub_word_filter(word)
        related_processor.frequency_filter(thresholds=(50, 10000), filter_corpus=freq_filter_corpus)
        words = related_processor.corpus.words
        print(word, words)
