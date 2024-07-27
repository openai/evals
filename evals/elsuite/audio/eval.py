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

from collections import Counter

from typing import Any, List, Optional
import evals
import evals.metrics
from evals.api import CompletionFn
from evals.record import RecorderBase
from evals.elsuite.modelgraded.classify_utils import classify

logger = logging.getLogger(__name__)
DEFAULT_SAMPLE_RATE = 16000


class Sample(BaseModel):
    audio: Any
    transcript: str
    expected: str
    context: Optional[str] = None


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
        prompt = self.build_prompt(sample)
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
        return self.compute_metrics(sample, sampled)

    def run(self, recorder: RecorderBase):
        samples = self._get_dataset()
        self.eval_all_samples(recorder, samples)

    def _get_dataset(self) -> List[Sample]:
        parsed = urlparse(self.dataset)
        query = parse_qs(parsed.query)
        query = {k: v[0] for k, v in query.items()}
        dataset = load_dataset(parsed.netloc + parsed.path, **query).cast_column("audio", Audio(sampling_rate=DEFAULT_SAMPLE_RATE))
        return [self.load_sample(sample) for sample in dataset]


class MatchAudioTask(AudioTask):
    def run(self, recorder: RecorderBase):
        super().run(recorder)
        events = recorder.get_events("match")
        expected = list(map(lambda e: e.data["expected"], events))
        sampled = list(map(lambda e: e.data["sampled"], events))
        return self.compute_corpus_metrics(events, expected, sampled)

    # Default implementation, if no additional metrics are added.
    def compute_corpus_metrics(self, events, expected: List[str], sampled: List[str]):
        return {"accuracy": evals.metrics.get_accuracy(events)}


class ModelGradedAudioTask(AudioTask):
    def __init__(self, completion_fns: list[CompletionFn], dataset: str, eval_completion_fn: CompletionFn, modelgraded_spec: str, modelgraded_spec_args: Optional[dict[str, dict[str, str]]] = None, *args, **kwargs):
        super().__init__(completion_fns, dataset, *args, **kwargs)
        self.eval_completion_fn = evals.registry.Registry().make_completion_fn(eval_completion_fn)
        self.eval_completion_kwargs = {"max_tokens": 1024}
        self.mg = self.registry.get_modelgraded_spec(modelgraded_spec)
        self.modelgraded_spec_args = modelgraded_spec_args or {}

    def compute_metrics(self, sample: Sample, sampled: str):
        completions = {"completion": sampled}
        test_sample = {"input": self.build_prompt(sample) + f"\n{sample.transcript}"}
        choice, info = classify(
            mg=self.mg,
            completion_fn=self.eval_completion_fn,
            completion_kwargs=self.eval_completion_kwargs,
            format_kwargs={**completions, **test_sample, **self.modelgraded_spec_args},
        )
        evals.record.record_metrics(choice=choice, score=info["score"])

    def run(self, recorder: RecorderBase):
        super().run(recorder)
        record_metrics = {}
        events = recorder.get_metrics()
        choices = [m["choice"] for m in events]
        counts = dict(Counter(choices))
        record_metrics.update({f"counts/{k}": v for k, v in counts.items()})
        scores = [m["score"] for m in events if m["score"] is not None]
        if scores:
            record_metrics["score"] = sum(scores) / len(scores)
        return record_metrics


class Transcribe(MatchAudioTask):
    def load_sample(self, row):
        return Sample(audio=row["audio"]["array"], transcript=row["text"], expected=row["text"])

    def build_prompt(self, sample: Sample):
        return "Repeat after me in English:"

    def compute_metrics(self, sample: Sample, sampled):
        expected = sample.expected
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


class Translate(MatchAudioTask):
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

    def load_sample(self, row):
        return Sample(audio=row["audio"]["array"], transcript=row["sentence"], expected=row["translation"])

    def build_prompt(self, sample: Sample):
        return "Translate the following into %s, without any explanation:" % self.target_language

    def compute_metrics(self, sample: Sample, sampled: str):
        expected = sample.expected
        score = self.bleu.sentence_score(sampled, [expected]).score
        evals.record.record_metrics(sacrebleu_sentence_score=score)
        match = score > 30
        if score is not None:
            evals.record.record_match(match, expected=expected, sampled=sampled, sacrebleu_sentence_score=score)
        return match

    def compute_corpus_metrics(self, events, expected: List[str], sampled: List[str]):
        refs = [[e] for e in expected]
        return {"accuracy": evals.metrics.get_accuracy(events), "sacrebleu_score": self.bleu.corpus_score(sampled, refs).score}


class SpokenER(MatchAudioTask):
    # Note, this is specific to the individual SpokenER dataset and a more general approach will be needed.
    EMOTIONS = [
        "anger",
        "joy",
        "neutral",
        "sadness",
    ]

    def load_sample(self, row):
        return Sample(audio=row["audio"]["array"], transcript="", expected=self.EMOTIONS[row["label"]])

    def build_prompt(self, sample: Sample):
        return "Respond in a single word which of these emotions (%s) best matches the following:" % ", ".join(self.EMOTIONS)

    def compute_metrics(self, sample: Sample, sampled: str):
        expected = sample.expected
        normalized = sampled.strip().rstrip(string.punctuation).lower()
        match = normalized == expected
        evals.record.record_match(match, expected=expected, sampled=sampled)
        return match


class SpokenBoolQ(MatchAudioTask):
    def load_sample(self, row):
        return Sample(audio=row["audio"]["array"], transcript="question", expected=str(row["answer"]).lower(), context=row["passage"])

    def build_prompt(self, sample: Sample):
        return f'Context:\n{sample.context}\n\nAnswer the following question with either "True" or "False", without any explanation:'

    def compute_metrics(self, sample: Sample, sampled: str):
        expected = sample.expected
        normalized = sampled.strip().rstrip(string.punctuation).lower()
        match = normalized == expected
        evals.record.record_match(match, expected=expected, sampled=sampled)
        return match


class SpokenQA(ModelGradedAudioTask):
    def load_sample(self, row):
        expected = row["answers"][0]["text"] if not row["is_impossible"] else "the question is impossible to answer"
        return Sample(audio=row["audio"]["array"], transcript=row["question"], expected=expected, context=row["context"])

    def build_prompt(self, sample: Sample):
        return f"Context:\n{sample.context}\n\nAnswer the following question:"
