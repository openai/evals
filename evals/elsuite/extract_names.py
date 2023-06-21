import evals
import evals.metrics
import random


class ExtractNames(evals.Eval):
    def __init__(self, test_jsonl, **kwargs):
        super().__init__(**kwargs)
        self.test_jsonl = test_jsonl

    def run(self, recorder):
        test_samples = evals.get_jsonl(self.test_jsonl)
        self.eval_all_samples(recorder, test_samples)
        # Record overall metrics
        return {
            "accuracy": evals.metrics.get_accuracy(recorder.get_events("match")),
        }

    def eval_sample(self, test_sample, rng: random.Random):
        prompt = test_sample["input"]
        result = self.completion_fn(
            prompt=prompt,
            max_tokens=25
        )
        sampled = result.get_completions()[0]
        evals.record_and_check_match(
            prompt,
            sampled,
            expected=test_sample["ideal"]
        )
