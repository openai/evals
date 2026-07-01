import json
import re
from typing import Any, Optional

import evals
import evals.metrics
from evals.api import CompletionFn
from evals.prompt.base import is_chat_prompt
from evals.record import record_metrics


def normalize_answer(answer: str) -> str:
    return answer.strip().lower()


def compare_answers(predicted: str, expected: str) -> bool:
    predicted_norm = normalize_answer(predicted)
    expected_norm = normalize_answer(expected)
    return predicted_norm == expected_norm or expected_norm in predicted_norm


def extract_confidence(text: str) -> Optional[float]:
    patterns = [
        r'confidence:?\s*(\d+(?:\.\d+)?)\s*%',
        r'(\d{1,3})\s*%\s*confiden',
        r'confiden[ct][ce]\s*(?:level|score)?:?\s*(\d+(?:\.\d+)?)\s*%?',
        r'probability:?\s*(\d+(?:\.\d+)?)\s*%',
        r'(\d{1,3})\s*%\s*certain',
        r'(\d{1,3})\s*%\s*sure',
        r'certainty:?\s*(\d+(?:\.\d+)?)\s*%',
    ]
    text_lower = text.lower()
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return float(match.group(1)) / 100.0
    return None


def extract_final_answer(text: str) -> str:
    markers = [
        r'(?:final\s+)?answer:?\s*(.+)',
        r'(?:revised\s+)?conclusion:?\s*(.+)',
        r'final\s+(?:result|response):?\s*(.+)',
        r'(?:correct\s+)?solution:?\s*(.+)',
    ]
    text_lower = text.lower()
    for marker in markers:
        match = re.search(marker, text_lower)
        if match:
            return match.group(1).strip()
    return text.strip()


def is_confirmation(text: str) -> bool:
    text_lower = text.lower()
    confirmations = [
        "your answer is correct",
        "your answer is already correct",
        "the answer remains",
        "i keep my answer",
        "i stand by my answer",
        "no revision needed",
        "my answer is correct",
        "the answer stays",
        "my original answer",
    ]
    return any(p in text_lower for p in confirmations)


CRITIQUE_PROMPTS = {
    "standard": """Question:
{question}

Your answer:
{answer}

Review your reasoning carefully.
Identify any assumptions, mistakes, or weaknesses in your answer.
Revise your answer if necessary.
Provide your final answer.
Also state your confidence as a percentage (e.g., Confidence: 85%).""",

    "assumptions": """Question:
{question}

Your answer:
{answer}

Before revising, list every assumption you made when solving this problem.
Then, for each assumption, evaluate whether it is justified.
If any assumptions are unwarranted, revise your answer.
Provide your final answer.
Also state your confidence as a percentage (e.g., Confidence: 85%).""",

    "disprove": """Question:
{question}

Your answer:
{answer}

Now, attempt to prove your answer WRONG.
Find counterarguments, edge cases, or alternative interpretations.
If your original answer survives this attack, keep it.
If not, provide the corrected answer.
Provide your final answer.
Also state your confidence as a percentage (e.g., Confidence: 85%).""",
}


class SelfCorrectionEval(evals.Eval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        samples_jsonl: str,
        *args,
        max_tokens: int = 500,
        num_few_shot: int = 0,
        few_shot_jsonl: str = None,
        critique_style: str = "standard",
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert len(completion_fns) == 1, "SelfCorrectionEval only supports one completion fn"
        self.max_tokens = max_tokens
        self.samples_jsonl = samples_jsonl
        self.num_few_shot = num_few_shot
        self.critique_style = critique_style
        if critique_style not in CRITIQUE_PROMPTS:
            raise ValueError(f"Unknown critique_style: {critique_style}. Options: {list(CRITIQUE_PROMPTS.keys())}")
        if self.num_few_shot > 0:
            assert few_shot_jsonl is not None, "few shot requires few shot sample dataset"
            self.few_shot_jsonl = few_shot_jsonl
            self.few_shot = evals.get_jsonl(self._prefix_registry_path(self.few_shot_jsonl))

    def eval_sample(self, sample: Any, rng):
        assert isinstance(sample, dict), "sample must be a dict"
        assert "input" in sample, "sample must have an 'input' key"
        assert "ideal" in sample, "sample must have an 'ideal' key"

        question = sample["input"]
        ideal = sample["ideal"]

        prompt = question
        if self.num_few_shot > 0:
            assert is_chat_prompt(question), "few shot requires chat prompt"
            prompt = question[:-1]
            for s in self.few_shot[: self.num_few_shot]:
                prompt += s["sample"]
            prompt += question[-1:]

        result1 = self.completion_fn(prompt=prompt, temperature=0.0, max_tokens=self.max_tokens)
        answer_1 = result1.get_completions()[0]
        confidence_1 = extract_confidence(answer_1)
        final_1 = extract_final_answer(answer_1)

        correct_1 = compare_answers(final_1, ideal)

        critique_template = CRITIQUE_PROMPTS[self.critique_style]
        critique_prompt = critique_template.format(question=question, answer=answer_1)
        if self.num_few_shot > 0 and is_chat_prompt(question):
            critique_prompt = [
                {"role": "system", "content": "You are a helpful assistant that reviews and improves answers."},
                {"role": "user", "content": critique_prompt}
            ]

        result2 = self.completion_fn(prompt=critique_prompt, temperature=0.0, max_tokens=self.max_tokens)
        answer_2 = result2.get_completions()[0]
        confidence_2 = extract_confidence(answer_2)
        final_2 = extract_final_answer(answer_2)

        if is_confirmation(answer_2):
            correct_2 = correct_1
            final_2 = final_1

        correct_2 = compare_answers(final_2, ideal) if not is_confirmation(answer_2) else correct_1

        revised = normalize_answer(final_1) != normalize_answer(final_2)
        improved = not correct_1 and correct_2
        degraded = correct_1 and not correct_2

        confidence_shift = None
        if confidence_1 is not None and confidence_2 is not None:
            confidence_shift = confidence_2 - confidence_1

        record_metrics(
            first_correct=correct_1,
            revised_correct=correct_2,
            improved=improved,
            degraded=degraded,
            stable=(correct_1 and correct_2),
            revised=revised,
            confidence_1=confidence_1,
            confidence_2=confidence_2,
            confidence_shift=confidence_shift,
        )

    def get_samples(self):
        if self.samples_jsonl is None:
            raise ValueError("samples_jsonl must be provided")
        samples = []
        with open(self.samples_jsonl, 'r') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return samples

    def run(self, recorder):
        samples = self.get_samples()
        self.eval_all_samples(recorder, samples)

        events = recorder.get_events("metrics")
        if not events:
            return self._empty_result()

        total = len(events)
        first_correct = sum(1 for e in events if e.data.get("first_correct", False))
        revised_correct = sum(1 for e in events if e.data.get("revised_correct", False))
        improved = sum(1 for e in events if e.data.get("improved", False))
        degraded = sum(1 for e in events if e.data.get("degraded", False))
        stable = sum(1 for e in events if e.data.get("stable", False))
        revised_count = sum(1 for e in events if e.data.get("revised", False))

        first_accuracy = first_correct / total
        revised_accuracy = revised_correct / total
        improvement_rate = improved / total
        degradation_rate = degraded / total
        stability_rate = stable / total
        net_gain = revised_accuracy - first_accuracy

        revision_rate = revised_count / total
        useful_revision_rate = improved / revised_count if revised_count > 0 else 0.0
        harmful_revision_rate = degraded / revised_count if revised_count > 0 else 0.0
        correction_efficiency = improvement_rate / revision_rate if revision_rate > 0 else 0.0

        confidence_shifts = [
            e.data["confidence_shift"] for e in events
            if e.data.get("confidence_shift") is not None
        ]
        avg_confidence_shift = sum(confidence_shifts) / len(confidence_shifts) if confidence_shifts else 0.0

        conf_1s = [e.data["confidence_1"] for e in events if e.data.get("confidence_1") is not None]
        conf_2s = [e.data["confidence_2"] for e in events if e.data.get("confidence_2") is not None]
        avg_confidence_1 = sum(conf_1s) / len(conf_1s) if conf_1s else 0.0
        avg_confidence_2 = sum(conf_2s) / len(conf_2s) if conf_2s else 0.0

        return {
            "first_accuracy": first_accuracy,
            "revised_accuracy": revised_accuracy,
            "improvement_rate": improvement_rate,
            "degradation_rate": degradation_rate,
            "stability_rate": stability_rate,
            "net_gain": net_gain,
            "revision_rate": revision_rate,
            "useful_revision_rate": useful_revision_rate,
            "harmful_revision_rate": harmful_revision_rate,
            "correction_efficiency": correction_efficiency,
            "avg_confidence_1": avg_confidence_1,
            "avg_confidence_2": avg_confidence_2,
            "avg_confidence_shift": avg_confidence_shift,
        }

    def _empty_result(self):
        return {
            "first_accuracy": 0.0,
            "revised_accuracy": 0.0,
            "improvement_rate": 0.0,
            "degradation_rate": 0.0,
            "stability_rate": 0.0,
            "net_gain": 0.0,
            "revision_rate": 0.0,
            "useful_revision_rate": 0.0,
            "harmful_revision_rate": 0.0,
            "correction_efficiency": 0.0,
            "avg_confidence_1": 0.0,
            "avg_confidence_2": 0.0,
            "avg_confidence_shift": 0.0,
        }