import evals
import evals.metrics
from evals.api import CompletionFn
from evals.record import RecorderBase

from .sage_eval import SageEval


class SageMultiChoice(SageEval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        samples_path: str,
        system_prompt: str,
        prompt_template: str,
        choice_ids: list[str],
        choice_descriptions: list[str],
        answer_id: str,
        *args,
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert len(completion_fns) == 1, "SageMultiChoice only supports one completion fn"

        self.samples_path = samples_path
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self.choice_ids = choice_ids
        self.choice_descriptions = choice_descriptions
        self.answer_id = answer_id

    def eval_sample(self, sample, rng):
        choice_prompt = ""
        for id, desc in zip(self.choice_ids, self.choice_descriptions):
            choice_prompt += f"{id}: {self.format_annotated_string(desc, sample)}\n"

        prompt = [
            {
                "role": "system",
                "message": self.format_annotated_string(self.system_prompt, sample),
            },
            {
                "role": "prompt",
                "message": self.format_annotated_string(self.prompt_template, sample)
                + "\n"
                + choice_prompt,
            },
        ]
        print()
        print("Sample", sample)
        print("Prompt", prompt)
        print("Expected", self.format_annotated_string(self.answer_id, sample))

        result = self.completion_fn(
            prompt=prompt,
            temperature=0.0,
            max_tokens=1,
        )
        sampled = result.get_completions()[0]

        evals.record_and_check_match(
            prompt=prompt,
            sampled=sampled,
            expected=self.format_annotated_string(self.answer_id, sample),
        )

    def run(self, recorder: RecorderBase):
        samples = self.get_csv(self.samples_path)
        self.eval_all_samples(recorder, samples)
        return {
            "accuracy": evals.metrics.get_accuracy(recorder.get_events("match")),
        }
