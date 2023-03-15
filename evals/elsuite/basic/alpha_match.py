import evals
import evals.metrics
from evals.record import RecorderBase


class AlphaMatch(evals.Eval):
    def __init__(
        self,
        model_specs: evals.ModelSpecs,
        samples_jsonl: str,
        *args,
        max_tokens: int = 500,
        **kwargs,
    ):
        super().__init__(model_specs, *args, **kwargs)
        self.max_tokens = max_tokens
        self.samples_jsonl = samples_jsonl

    def eval_sample(self, test_sample, rng):
        prompt, correct_answer = test_sample["input"], test_sample["ideal"]
        generated_answer = evals.sample_freeform(
            self.model_spec,
            prompt,
            temperature=0.0,
            max_tokens=16,
        )

        get_alphas = lambda in_str: "".join(c for c in in_str.lower() if c.isalpha())

        is_match = get_alphas(generated_answer) == get_alphas(correct_answer)

        evals.record.record_match(is_match, expected=correct_answer, picked=generated_answer)

    def run(self, recorder: RecorderBase):
        samples = evals.get_jsonl(self.samples_jsonl)
        self.eval_all_samples(recorder, samples)

        events = recorder.get_events("match")
        return {
            "accuracy": evals.metrics.get_accuracy(events),
        }
