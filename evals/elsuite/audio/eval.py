import base64
import jiwer
import soundfile as sf
import logging
import string
from io import BytesIO
from urllib.parse import parse_qs, urlparse

from datasets import load_dataset, Audio
from pydantic import BaseModel
from sacrebleu.metrics.bleu import BLEU


from typing import Any, List
import evals
import evals.metrics
from evals.api import CompletionFn
from evals.record import RecorderBase

logger = logging.getLogger(__name__)
DEFAULT_SAMPLE_RATE = 16000


class Sample(BaseModel):
    audio: Any
    expected: str


class AudioTask(evals.Eval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        dataset: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        *args,
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert len(completion_fns) == 1, "Audio tasks only support one completion fn"
        self.dataset = dataset
        self.temperature = temperature
        self.max_tokens = max_tokens

    def eval_sample(self, sample: Sample, rng):
        assert isinstance(sample, Sample)
        prompt = self.get_prompt()
        with BytesIO() as buffer:
            sf.write(buffer, sample.audio, DEFAULT_SAMPLE_RATE, format="WAV", subtype="PCM_16")
            base_64_wav = base64.b64encode(buffer.getvalue()).decode()

        try:
            result = self.completion_fn(
                prompt=[
                    {"role": "system", "content": "You are an audio language model with the ability to understand human speech."},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:audio/x-wav;base64,{base_64_wav}",
                                },
                            },
                        ],
                    },
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            sampled = result.get_completions()[0]
        except Exception as e:
            logging.info("Sampling failed!")
            logging.info(sample)
            logging.info(f"Prompt: {prompt}")
            logging.info(f"Error: {str(e)}")
            sampled = "ERROR: " + str(e)
        return self.compute_metrics(sample.expected, sampled)

    def run(self, recorder: RecorderBase):
        samples = self._get_dataset()
        self.eval_all_samples(recorder, samples)
        events = recorder.get_events("match")
        expected = list(map(lambda e: e.data["expected"], events))
        sampled = list(map(lambda e: e.data["sampled"], events))
        return self.compute_corpus_metrics(events, expected, sampled)

    # Default implementation, if no additional metrics are added.
    def compute_corpus_metrics(self, events, expected: List[str], sampled: List[str]):
        return {"accuracy": evals.metrics.get_accuracy(events)}

    def _get_dataset(self) -> List[Sample]:
        parsed = urlparse(self.dataset)
        query = parse_qs(parsed.query)
        query = {k: v[0] for k, v in query.items()}
        dataset = load_dataset(parsed.netloc + parsed.path, **query).cast_column("audio", Audio(sampling_rate=DEFAULT_SAMPLE_RATE))
        return [Sample(expected=self.get_expected(sample), audio=sample["audio"]["array"]) for sample in dataset]


class Transcribe(AudioTask):
    def get_prompt(self):
        return "Repeat after me in English:"

    def get_expected(self, row):
        return row["text"]

    def compute_metrics(self, expected, sampled):
        score = self._compute_wer(expected, sampled)
        evals.record.record_metrics(wer=score)
        match = score < 0.1
        evals.record.record_match(match, expected=expected, sampled=sampled, wer=score)
        return match

    def compute_corpus_metrics(self, events, expected, sampled):
        return {"accuracy": evals.metrics.get_accuracy(events), "wer": self._compute_wer(expected, sampled)}

    def _compute_wer(self, expected, sampled):
        transform = jiwer.Compose(
            [
                jiwer.RemovePunctuation(),
                jiwer.ToLowerCase(),
                jiwer.RemoveWhiteSpace(replace_by_space=True),
                jiwer.Strip(),
                jiwer.ReduceToListOfListOfWords(),
            ]
        )
        output = jiwer.process_words(expected, sampled, reference_transform=transform, hypothesis_transform=transform)
        return output.wer


class Translate(AudioTask):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        dataset: str,
        target_language: str,
        *args,
        **kwargs,
    ):
        super().__init__(completion_fns, dataset, *args, **kwargs)
        self.target_language = target_language
        self.bleu = BLEU(effective_order=True)

    def get_prompt(self):
        return "Translate the following into %s, without any explanation:" % self.target_language

    def get_expected(self, row):
        return row["translation"]

    def compute_metrics(self, expected: str, sampled: str):
        score = self.bleu.sentence_score(sampled, [expected]).score
        evals.record.record_metrics(sacrebleu_sentence_score=score)
        match = score > 30
        if score is not None:
            evals.record.record_match(match, expected=expected, sampled=sampled, sacrebleu_sentence_score=score)
        return match

    def compute_corpus_metrics(self, events, expected: List[str], sampled: List[str]):
        refs = [[e] for e in expected]
        return {"accuracy": evals.metrics.get_accuracy(events), "sacrebleu_score": self.bleu.corpus_score(sampled, refs).score}


class SpokenQA(AudioTask):
    def get_prompt(self):
        return "Answer the following question:"

    def get_expected(self, row):
        return row["answer"]

    def compute_metrics(self, expected: str, sampled: str):
        assert isinstance(sample, Sample)
        prompt = "Transcribe the following audio exactly:\n"
        sampled = self._sampled(prompt, sample)
        correct_answer = sample.text
        score = compute_wer(correct_answer, sampled)
        evals.record.record_metrics(wer=score)
        match = score < 0.1
        evals.record.record_match(match, expected=correct_answer, sampled=sampled, wer=score)
        return match

    def compute_corpus_metrics(self, events, expected: List[str], sampled: List[str]):
        return {"accuracy": evals.metrics.get_accuracy(events), "wer": compute_wer(expected, sampled)}


class SpokenER(AudioTask):
    # Note, this is specific to the individual SpokenER dataset and a more general approach will be needed.
    EMOTIONS = [
        "anger",
        "joy",
        "neutral",
        "sadness",
    ]

    def get_prompt(self):
        return "Choose which of these emotions (%s) best matches the following, without any explanation:" % ", ".join(self.EMOTIONS)

    def get_expected(self, row):
        return self.EMOTIONS[row["label"]]

    def compute_metrics(self, expected: str, sampled: str):
        normalized = sampled.strip().rstrip(string.punctuation).lower()
        match = normalized == expected
        evals.record.record_match(match, expected=expected, sampled=sampled)
        return match
