from typing import Any

import evals
import evals.metrics
from evals.elsuite import utils
from evals.prompt.base import is_chat_prompt


class Match(evals.Eval):
    def __init__(
        self,
        model_specs: evals.ModelSpecs,
        samples_jsonl: str,
        *args,
        max_tokens: int = 500,
        num_few_shot: int = 0,
        few_shot_jsonl: str = None,
        completion_fn: utils.CompletionFn = evals.completion_query,
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
        self._completion_fn = completion_fn

    def eval_sample(self, sample: Any, *_):
        prompt = sample["input"]
        if self.num_few_shot > 0:
            assert is_chat_prompt(sample["input"]), "few shot requires chat prompt"
            prompt = sample["input"][:-1]
            for s in self.few_shot[: self.num_few_shot]:
                prompt += s["sample"]
            prompt += sample["input"][-1:]

        # TODO(hwc): is there a case where we want to use `result` other than "choices"?
        result, actual_prompt, metadata = self._completion_fn(
            prompt=prompt,
            temperature=0.0,
            model_spec=self.model_spec,
        )
        choice = result["choices"][0]
        sampled = choice["text"].strip() if self.model_spec.strip_completion else choice["text"]
        return evals.record_and_check_match(
                prompt=actual_prompt,
                sampled=sampled,
                expected=sample["ideal"],
                metadata=metadata
        )

    def run(self, recorder):
        samples= self.get_samples()
        self.eval_all_samples(recorder, samples)
        events = recorder.get_events("match")
        return {
            "accuracy": evals.metrics.get_accuracy(events),
        }
