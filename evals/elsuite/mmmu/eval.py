import ast
import base64
import logging
from io import BytesIO
from typing import Optional, Union
from urllib.parse import parse_qs, urlparse

from datasets import load_dataset
from PIL import Image
from pydantic import BaseModel

import evals
import evals.metrics
from evals.api import CompletionFn
from evals.formatting import make_abc
from evals.record import RecorderBase, record_match

logger = logging.getLogger(__name__)


class Sample(BaseModel):
    question: str
    answers: list[str]
    label: Union[int, str]
    question_type: str
    image_1: Optional[Image.Image]
    image_2: Optional[Image.Image]
    image_3: Optional[Image.Image]
    image_4: Optional[Image.Image]
    image_5: Optional[Image.Image]
    image_6: Optional[Image.Image]
    image_7: Optional[Image.Image]

    class Config:
        arbitrary_types_allowed = True


def get_dataset(url: str) -> list[Sample]:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    query = {k: v[0] for k, v in query.items()}

    dataset = load_dataset("mmmu/mmmu", **query)

    return [
        Sample(
            question=sample["question"],
            answers=ast.literal_eval(sample["options"]),
            label=(
                ord(sample["answer"]) - ord("A")
                if sample["question_type"] == "multiple-choice"
                else sample["answer"]
            ),
            question_type=sample["question_type"],
            image_1=sample["image_1"],
            image_2=sample["image_2"],
            image_3=sample["image_3"],
            image_4=sample["image_4"],
            image_5=sample["image_5"],
            image_6=sample["image_6"],
            image_7=sample["image_7"],
        )
        for sample in dataset
    ]


class MMMU(evals.Eval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        dataset: str,
        subject: str,
        *args,
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert len(completion_fns) == 1, "MMMU only supports one completion fn"
        self.dataset = dataset
        self.subject = subject

    def eval_sample(self, sample: Sample, rng):
        assert isinstance(sample, Sample)

        if sample.question_type == "multiple-choice":
            options, correct_answer = make_abc(
                answers=sample.answers,
                correct_idx=sample.label,
                rng=rng,
            )
            prompt = sample.question + "\n" + options
            system_prompt = f'You are an expert in {self.subject} whose job is to answer questions from the user using images. First, reason about the correct answer. Then write the answer in the following format where X is exactly one of A,B,C,D: "ANSWER: X". If you are uncertain of the correct answer, guess the most likely one.'
        else:
            correct_answer = sample.label
            prompt = sample.question
            system_prompt = f'You are an expert in {self.subject} whose job is to answer questions from the user using images. First, reason about the correct answer. Then write the answer in the following format where X is only the answer and nothing else: "ANSWER: X"'

        images = [
            image
            for image in [
                sample.image_1,
                sample.image_2,
                sample.image_3,
                sample.image_4,
                sample.image_5,
                sample.image_6,
                sample.image_7,
            ]
            if image is not None
        ]

        base_64_images = []
        for image in images:
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue())
            base_64_images.append(img_str.decode())

        try:
            result = self.completion_fn(
                prompt=[
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": system_prompt,
                            },
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ]
                        + [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base_64_image}",
                                },
                            }
                            for base_64_image in base_64_images
                        ],
                    },
                ],
                temperature=0.0,
                max_tokens=4096,
            )
            sampled = result.get_completions()[0]
        except Exception as e:
            logging.info("Sampling failed!")
            logging.info(sample)
            logging.info(f"Prompt: {prompt}")
            logging.info(f"Error: {str(e)}")
            sampled = "ERROR: " + str(e)

        match = sampled.find(f"ANSWER: {correct_answer}") != -1

        if not match and sampled.find("ANSWER") == -1 and sample.question_type == "multiple-choice":
            # The model didn't answer anything, so randomly pick an answer
            # This matches the behavior described in section 4.1 of the MMMU paper: https://arxiv.org/pdf/2311.16502.pdf
            logging.info("No answer found for multiple choice so picking a random answer.")
            answer_idx = rng.randint(0, len(sample.answers) - 1)
            answer_letter = chr(ord("A") + answer_idx)
            match = correct_answer == answer_letter

        record_match(
            match,
            expected=correct_answer,
            picked=(correct_answer if match else None),
            sampled=sampled,
        )

    def run(self, recorder: RecorderBase):
        samples = get_dataset(self.dataset)
        self.eval_all_samples(recorder, samples)
        return {
            "accuracy": evals.metrics.get_accuracy(recorder.get_events("match")),
        }
