import random

import evals
import evals.metrics
from evals.record import record_match, record_sampling


class CharacterLimit(evals.Eval):
    def __init__(self, samples_jsonl, **kwargs):
        super().__init__(**kwargs)
        self.samples_jsonl = samples_jsonl

    def run(self, recorder):
        test_samples = evals.get_jsonl(self.samples_jsonl)
        self.eval_all_samples(recorder, test_samples)

        # Record overall metrics
        return {
            "accuracy": evals.metrics.get_accuracy(recorder.get_events("match")),
        }

    def eval_sample(self, test_sample, rng: random.Random):
        prompt = []

        for i, sample in enumerate([test_sample]):
            prompt += [{"role": "user", "content": sample[0]["content"]}]

        result, actual_prompt, metadata = evals.completion_query(
            prompt=prompt,
            temperature=0.0,
            model_spec=self.model_spec,
        )

        choice = result["choices"][0]["text"]
        picked = None
        expected = sample[1]["content"]
        sampled = None

        result = {
            "prompt": actual_prompt,
            "sampled": sampled,
            "options": None,
            "picked": picked,
        }

        # check if the number of characters in result string is within the expected range
        limit = expected.split("-")
        lower = int(limit[0])
        upper = int(limit[1])
        match = len(choice) >= lower and len(choice) <= upper

        result["expected"] = expected
        result["match"] = match
        result["metadata"] = metadata

        record_sampling(**result)
        record_match(match, expected=expected, picked=picked, sampled=sampled)
