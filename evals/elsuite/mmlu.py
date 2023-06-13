from typing import Optional
from pydantic import BaseModel
import evals
import evals.metrics
from evals.api import CompletionFn
from evals.formatting import make_abc
from evals.record import RecorderBase
from datasets import load_dataset

class Sample(BaseModel):
    question: str
    answers: list[str]
    label: int
    category: str

MMLU_CATEGORIES = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']

class MMLU(evals.Eval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        split: str,
        *args,
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert len(completion_fns) == 1, "MMLU only supports one completion fn"
        self.split = split

    def eval_sample(self, sample, rng):
        assert isinstance(sample, Sample)

        options, correct_answer = make_abc(
            answers=sample.answers,
            correct_idx=sample.label,
            rng=rng,
        )

        prompt = "Please answer with the letter of the correct answer." + "\n\n" + sample.question + "\n" + options
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
            category=sample.category,
        )

    def get_samples(self):
        samples = []
        for category in MMLU_CATEGORIES:
            dataset = load_dataset("hendrycks_test", name=category, split=self.split)
            for sample in dataset:
                samples.append(Sample(
                    question=sample["question"],
                    answers=sample["choices"],
                    label=sample["answer"],
                    category=category,
                ))
        return samples


    def run(self, recorder: RecorderBase):
        samples = self.get_samples()
        self.eval_all_samples(recorder, samples)

        # categorize accuracies
        events_by_category: dict[str, list] = {}
        events = recorder.get_events("match")
        for event in events:
            category = event.data["category"]
            if category not in events_by_category:
                events_by_category[category] = []
            events_by_category[category].append(event)

        # compute metrics per category
        category_metrics = {}
        category_accuracies = []
        category_weights = []
        for category, events in events_by_category.items():
            accuracy = evals.metrics.get_accuracy(events)
            count = len(events)
            category_accuracies.append(accuracy)
            category_weights.append(count)
            category_metrics.update({
                f"{category}/accuracy": accuracy,
                f"{category}/count": count,
                f"{category}/bootstrap_std": evals.metrics.get_bootstrap_accuracy_std(events),
            })

        # compute overall metrics
        overall_metrics = {
            "accuracy": evals.metrics.get_weighted_mean(category_accuracies, category_weights),
        }

        return {
            **overall_metrics,
            **category_metrics,
        }
