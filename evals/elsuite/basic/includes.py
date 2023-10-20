from typing import Any

import evals
import evals.metrics
from evals.api import CompletionFn
from evals.elsuite import utils


class Includes(evals.Eval):
    """
    For a model completion `a` and a reference list of correct answers `B`, the following eval 
    by default implements `any([(b in a) for b in B])`, as mentioned in `eval-templates.md`.
    However, it can be modified to account for terms that must be excluded: 
    For instance, given a reference list of answers-to-excluse `C`,
    the logic would become `any([(b in a) for b in B]) and not any([(c in a) for c in C])` -
    that is, if any excluded term is present, even if an expected term is present too, the answer will be `False`.
    """
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        samples_jsonl: str,
        ignore_case: bool = False,
        use_exclusions: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert len(completion_fns) == 1, "Includes only supports one completion fn"
        self.samples_jsonl = samples_jsonl
        self.ignore_case = ignore_case
        self.use_exclusions = use_exclusions
        """
        To modify the eval's logic to return False if an excluded term is present,
        set `use_exclusions: True` in your YAML and including an `"exclude"` key in your sample.
        See [`hanabi_clues`](root/evals/registry/evals/hanabi_clues.yaml) for a practical example.
        """

    def eval_sample(self, sample: Any, *_):
        assert isinstance(sample, dict), "sample must be a dict"
        assert "input" in sample, "sample must have an 'input' key"
        assert "ideal" in sample, "sample must have an 'ideal' key"
        if self.use_exclusions:
            assert "exclude" in sample, "sample must have an 'exclude' key if use_exclusions is set to be True"

        prompt = sample["input"]
        result = self.completion_fn(
            prompt=prompt,
        )
        sampled = result.get_completions()[0]

        ideal = sample["ideal"]
        if not isinstance(ideal, list):
            ideal = [ideal]

        assert isinstance(ideal, list) and all(
            isinstance(i, str) for i in ideal
        ), "ideal must be a list of strings"

        includes_answer = any(
            [utils.get_answer(sampled, ref, self.ignore_case) is not None for ref in ideal]
        )

        if self.use_exclusions:
            exclusions = sample["exclude"]
            
            # if any exclude_item is present, this makes the answer False, regardless of presence of any/all include items
            if any(exclude_item in sampled for exclude_item in exclusions):
                includes_answer = False
            
            excluded_item_found = True
        
        evals.record.record_match(
            includes_answer, expected=sample["ideal"], picked=sampled, sampled=sampled, excluded_item_found=excluded_item_found,
        )
        return includes_answer

    def run(self, recorder):
        samples = self.get_samples()
        self.eval_all_samples(recorder, samples)
        events = recorder.get_events("match")
        return {
            "accuracy": evals.metrics.get_accuracy(events),
            "boostrap_std": evals.metrics.get_bootstrap_accuracy_std(events),
        }
