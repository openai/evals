import random
import textwrap

import evals
import evals.metrics

class FamousShowsImages(evals.Eval):
    def __init__(self, train_jsonl, test_jsonl, train_samples_per_prompt=2, **kwargs):
        super().__init__(**kwargs)
        self.train_jsonl = train_jsonl
        self.test_jsonl = test_jsonl
        self.train_samples_per_prompt = train_samples_per_prompt

    def run(self, recorder):
        self.train_samples = evals.get_jsonl(self.train_jsonl)
        test_samples = evals.get_jsonl(self.test_jsonl)
        self.eval_all_samples(recorder, test_samples)

        return {
            "accuracy": evals.metrics.get_accuracy(recorder.get_events("match")),
        }
    def eval_sample(self, test_sample, rng: random.Random):
        stuffing = rng.sample(self.train_samples, self.train_samples_per_prompt)

        prompt = [
            {"role": "system", "content": "Solve the following math problems"},
        ]

        for i, sample in enumerate(stuffing + [test_sample]):
            if i < len(stuffing):
                prompt += [
                    {"role": "system", "content": sample[0]["content"], "name": "example_user"},
                    {"role": "system", "content": sample[1]["content"], "name": "example_assistant"},
                ]
            else:
                prompt += [{"role": "user", "content": sample[0]["content"]}]

        result = self.completion_fn(
            prompt=prompt,
            temperature=0.0,
        )
        sampled = result.get_completions()[0]

        return evals.record_and_check_match(
            prompt=prompt,
            sampled=sampled,
            expected=sample[1]["content"],
        )