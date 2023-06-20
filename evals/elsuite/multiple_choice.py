from typing import Optional
from urllib.parse import parse_qs, urlparse
from pydantic import BaseModel
import evals
import json
import evals.metrics
from evals.api import CompletionFn
from evals.formatting import make_abc
from evals.record import RecorderBase
from datasets import load_dataset
from evals.utils.embeddings import process_text, retrieve_context_from_db

class Sample(BaseModel):
    article: str
    question: str
    answers: list[str]
    label: int

def get_dataset(url: str) -> list[Sample]:

    with open(url, 'r') as f:
        data = [json.loads(line) for line in f]
    return [Sample(article = d['context'], question=d['question'], answers=d['answers'], label=d['label']) for d in data]


class MultipleChoice(evals.Eval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        dataset: str,
        *args,
        instructions: Optional[str] = "",
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert len(completion_fns) == 1, "MultipleChoice only supports one completion fn"
        self.dataset = dataset
        self.instructions = instructions

    def eval_sample(self, sample, rng):
        assert isinstance(sample, Sample)

        options, correct_answer = make_abc(
            answers=sample.answers,
            correct_idx=sample.label,
            rng=rng,
        )
        try:
            db = process_text(sample.article)
            context = retrieve_context_from_db(sample.question, db)
            prompt = self.instructions + "\nUsing the provided context, Please answer with the letter of the correct answer.Context:"+ context + "\n\n"+"The questions" + "\n\n" + sample.question + "\n" + options
            print(prompt)
            result = self.completion_fn(
                prompt=prompt,
                temperature=0.0,
                max_tokens=1,
            )
            sampled = result.get_completions()[0]
            print("Guess vs correct answer:", sampled, correct_answer)
            evals.record_and_check_match(
                prompt=prompt,
                sampled=sampled,
                expected=correct_answer,
            )
        except Exception as e:
            print("Error:", e)
            

    def run(self, recorder: RecorderBase):
        samples = get_dataset(self.dataset)
        self.eval_all_samples(recorder, samples)
        return {
            "accuracy": evals.metrics.get_accuracy(recorder.get_events("match")),
        }
