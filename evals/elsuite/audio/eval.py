import base64
import jiwer
import soundfile as sf
import logging
from io import BytesIO
from urllib.parse import parse_qs, urlparse

from datasets import load_dataset, Audio
from pydantic import BaseModel

# import numpy as np
from typing import Any
import evals
import evals.metrics
from evals.api import CompletionFn
from evals.record import RecorderBase

logger = logging.getLogger(__name__)
DEFAULT_SAMPLE_RATE = 16000


class Sample(BaseModel):
    audio: Any
    text: str


def get_dataset(url: str) -> list[Sample]:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    query = {k: v[0] for k, v in query.items()}
    dataset = load_dataset(parsed.netloc + parsed.path, **query).cast_column("audio", Audio(sampling_rate=DEFAULT_SAMPLE_RATE))
    return [Sample(text=sample["text"], audio=sample["audio"]["array"]) for sample in dataset]


def compute_wer(expected, sampled):
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


class Transcribe(evals.Eval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        dataset: str,
        *args,
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert len(completion_fns) == 1, "Transcribe only supports one completion fn"
        self.dataset = dataset

    def eval_sample(self, sample: Sample, rng):
        assert isinstance(sample, Sample)

        correct_answer = sample.text
        prompt = "Transcribe the following audio exactly:\n"
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
                temperature=0.7,
                max_tokens=4096,
            )
            sampled = result.get_completions()[0]
        except Exception as e:
            logging.info("Sampling failed!")
            logging.info(sample)
            logging.info(f"Prompt: {prompt}")
            logging.info(f"Error: {str(e)}")
            sampled = "ERROR: " + str(e)

        score = compute_wer(correct_answer, sampled)
        evals.record.record_metrics(wer=score)
        match = score < 0.1
        evals.record.record_match(match, expected=correct_answer, sampled=sampled, wer=score)
        return match

    def run(self, recorder: RecorderBase):
        samples = get_dataset(self.dataset)
        self.eval_all_samples(recorder, samples)
        events = recorder.get_events("match")

        expected = list(map(lambda e: e.data["expected"], events))
        sampled = list(map(lambda e: e.data["sampled"], events))
        score = compute_wer(expected, sampled)

        return {
            "accuracy": evals.metrics.get_accuracy(events),
            "wer": score,
        }
