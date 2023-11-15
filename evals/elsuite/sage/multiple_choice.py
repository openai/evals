from sage_eval import SageEval

import evals
import evals.metrics
from evals.api import CompletionFn
from evals.formatting import make_abc
from evals.record import RecorderBase


class SageMultiChoice(SageEval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        samples_path: str,
        samples_fieldnames: list[str],
        system_prompt: str,
        prompt_template: str,
        choices: list[str],
        answer: str,
        *args,
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert len(completion_fns) == 1, "SageMultiChoice only supports one completion fn"

        self.samples_path = samples_path
        self.samples_fieldnames = samples_fieldnames
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template

    def eval_sample(self, sample, rng):
        options, correct_answer = make_abc(
            answers=sample.answers,
            correct_idx=sample.label,
            rng=rng,
        )

        prompt = [
            {
                "role": "system",
                "message": self.format_prompt(self.system_prompt, sample),
            },
            {
                "role": "prompt",
                "message": self.format_prompt(self.prompt_template, sample),
            },
        ]

        result = self.completion_fn(
            prompt=prompt,
            temperature=0.0,
            max_tokens=1,
        )
        sampled = result.get_completions()[0]

        evals.record_and_check_match(
            prompt=prompt,
            sampled=sampled,
            expected=correct_answer,
        )

    def run(self, recorder: RecorderBase):
        samples = self.get_csv(self.samples_path, fieldnames=self.samples_fieldnames)
        self.eval_all_samples(recorder, samples)
        return {
            "accuracy": evals.metrics.get_accuracy(recorder.get_events("match")),
        }
