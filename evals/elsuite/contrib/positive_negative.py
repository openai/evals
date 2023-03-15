import evals
import evals.metrics
from evals.record import RecorderBase


class PositiveNegativeMatch(evals.Eval):
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
        sample_id = test_sample["sample_id"]
        prompt = test_sample["input"]
        correct_answers = test_sample["positive"]
        incorrect_answers = test_sample["negative"]

        generated_answer = evals.sample_freeform(
            self.model_spec,
            prompt,
            temperature=0.0,
            max_tokens=1024,
        )

        generated_answer = generated_answer.lower()

        # print("=== Question ===")
        # print(prompt)
        # print("=== Answer ===")
        # print(generated_answer)

        correct_pick = ""
        correct = False
        for correct_answer in correct_answers:
            if correct_answer in generated_answer:
                correct_pick = correct_answer
                correct = True
                break
        for incorrect_answer in incorrect_answers:
            if incorrect_answer in generated_answer:
                correct = False
                break

        evals.record.record_match(
            correct=correct, expected=correct_answers, picked=correct_pick, sample_id=sample_id
        )

    def run(self, recorder: RecorderBase):
        samples = evals.get_jsonl(self.samples_jsonl)
        self.eval_all_samples(recorder, samples)
        events = recorder.get_events("match")
        return {"accuracy": evals.metrics.get_accuracy(events)}
