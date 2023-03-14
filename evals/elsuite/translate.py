from typing import Any

from sacrebleu.metrics.bleu import BLEU

import evals
import evals.metrics
from evals.prompt.base import is_chat_prompt


class Translate(evals.Eval):
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

        self.bleu = BLEU(effective_order=True)

    def eval_sample(self, sample: Any, *_):
        prompt = sample["input"]
        expected = sample["ideal"]
        if self.num_few_shot > 0:
            assert is_chat_prompt(sample["input"]), "few shot requires chat prompt"
            prompt = sample["input"][:-1]
            for s in self.few_shot[: self.num_few_shot]:
                prompt += s["sample"]
            prompt += sample["input"][-1:]

        if isinstance(expected, tuple):
            expected = list(expected)
        elif not isinstance(expected, list):
            expected = [expected]

        sampled = evals.sample_freeform(self.model_spec, prompt, max_tokens=self.max_tokens)

        score = None
        if expected is not None:
            score = self.bleu.sentence_score(sampled, expected).score
            evals.record.record_metrics(sacrebleu_sentence_score=score)

            match = score > 30

            if score is not None:
                evals.record.record_match(
                    match, expected=expected, sampled=sampled, sacrebleu_sentence_score=score
                )
            return match

    def run(self, recorder):
        samples = evals.get_jsonl(self.samples_jsonl)
        self.eval_all_samples(recorder, samples)
        events = recorder.get_events("match")

        sampled = list(map(lambda e: e.data["sampled"], events))
        expected = list(map(lambda e: e.data["expected"], events))
        sacrebleu_score = BLEU().corpus_score(sampled, [expected]).score

        return {
            "accuracy": evals.metrics.get_accuracy(events),
            "sacrebleu_score": sacrebleu_score,
        }
