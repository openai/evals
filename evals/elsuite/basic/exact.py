from typing import Any
import json
import evals
import evals.metrics
from evals.prompt.base import is_chat_prompt
from evals.record import record_match, record_sampling

def strict_match(response: str, ideal: str) -> int:
    if response is None or ideal is None:
        return False

    if response == ideal:
        return True
    else:
        return False

class Match(evals.Eval):
    def __init__(
        self,
        model_specs: evals.ModelSpecs,
        samples_jsonl: str,
        *args,
        max_tokens: int = 500,
        num_few_shot: int = 0,
        few_shot_jsonl: str = None,
        **kwargs,
    ):
        super().__init__(model_specs, *args, **kwargs)
        self.max_tokens = max_tokens
        self.samples_jsonl = samples_jsonl
        self.num_few_shot = num_few_shot
        if self.num_few_shot > 0:
            assert few_shot_jsonl is not None, "few shot requires few shot sample dataset"
            self.few_shot_jsonl = few_shot_jsonl
            self.few_shot = evals.get_jsonl(self.few_shot_jsonl)

    def eval_sample(self, sample: Any, *_):
        prompt = sample["input"]
        if self.num_few_shot > 0:
            assert is_chat_prompt(sample["input"]), "few shot requires chat prompt"
            prompt = sample["input"][:-1]
            for s in self.few_shot[: self.num_few_shot]:
                prompt += s["sample"]
            prompt += sample["input"][-1:]

        print(prompt)

        result, actual_prompt, metadata = evals.completion_query(
        prompt=prompt,
        temperature=0.0,
        model_spec=self.model_spec)

        print(result)

        choice = result["choices"][0]

        score = strict_match(choice["text"], sample["ideal"])

        print(score)

        result = {
            "prompt": actual_prompt,
            "sampled": choice["text"],
            "options": None,
            "picked": choice["text"],
        }

        record_sampling(**result)
        record_match(score, expected=sample["ideal"], picked=choice["text"], sampled=choice["text"])

    def run(self, recorder):
        samples = evals.get_jsonl(self.samples_jsonl)
        self.eval_all_samples(recorder, samples)
        events = recorder.get_events("match")
        return {
            "accuracy": evals.metrics.get_accuracy(events),
        }