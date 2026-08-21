import importlib
from unittest.mock import call, patch

import pytest

import evals.record
from evals.elsuite.modelgraded.base import ModelGradedSpec
from evals.elsuite.modelgraded.classify import ModelBasedClassify
from evals.elsuite.utils import PromptFn


class FakeCompletionResult:
    def __init__(self, completions: list[str]):
        self.completions = completions

    def get_completions(self) -> list[str]:
        return self.completions


class FakeCompletionFn:
    def __init__(self, completions: list[str]):
        self.completions = completions
        self.calls: list[dict] = []

    def __call__(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        return FakeCompletionResult(self.completions)


def test_prompt_fn_sample_all_preserves_all_requested_completions() -> None:
    completion_fn = FakeCompletionFn(["first", "second", "third"])
    prompt_fn = PromptFn(
        "Question: {question}",
        completion_fn=completion_fn,
        max_tokens=32,
        temperature=0.7,
        n_samples=3,
    )

    completions, prompt = prompt_fn.sample_all(question="hello")

    assert completions == ["first", "second", "third"]
    assert prompt == "Question: hello"
    assert completion_fn.calls[0]["n"] == 3
    assert completion_fn.calls[0]["temperature"] == 0.7


def test_prompt_fn_call_keeps_single_completion_compatibility() -> None:
    completion_fn = FakeCompletionFn(["first", "second"])
    prompt_fn = PromptFn("hello", completion_fn=completion_fn, max_tokens=32)

    completion, prompt = prompt_fn()

    assert completion == "first"
    assert prompt == "hello"
    assert completion_fn.calls[0]["n"] == 1


def make_model_based_classify(completion_fn: FakeCompletionFn, n_samples: int):
    evaluator = ModelBasedClassify.__new__(ModelBasedClassify)
    evaluator.completion_fns = [completion_fn]
    evaluator.eval_completion_fn = object()
    evaluator.sample_kwargs = {"max_tokens": 32, "temperature": 0.8, "n_samples": n_samples}
    evaluator.eval_kwargs = {"max_tokens": 32}
    evaluator.metaeval = False
    evaluator.modelgraded_spec_args = {}
    evaluator.eval_type = "classify"
    evaluator.match_fn = None
    evaluator.multicomp_n = 1
    evaluator.n_samples = n_samples
    evaluator.mg = ModelGradedSpec(
        prompt="Grade {completion}",
        choice_strings=["A", "B"],
        input_outputs={"input": "completion"},
        eval_type="classify",
        choice_scores={"A": 1.0, "B": 0.0},
    )
    return evaluator


def test_model_based_classify_grades_each_sampled_completion() -> None:
    policy = FakeCompletionFn(["candidate 1", "candidate 2", "candidate 3"])
    evaluator = make_model_based_classify(policy, n_samples=3)
    classify_module = importlib.import_module("evals.elsuite.modelgraded.classify")
    seen_completions: list[str] = []

    def fake_classify(**kwargs):
        completion = kwargs["format_kwargs"]["completion"]
        seen_completions.append(completion)
        if completion == "candidate 2":
            return "B", {"score": 0.0}
        return "A", {"score": 1.0}

    with patch.object(classify_module, "classify", side_effect=fake_classify), patch.object(
        evals.record, "record_metrics"
    ) as record_metrics:
        choices = evaluator.eval_sample({"input": "question"}, None)

    assert choices == ["A", "B", "A"]
    assert seen_completions == ["candidate 1", "candidate 2", "candidate 3"]
    assert len(policy.calls) == 1
    assert policy.calls[0]["n"] == 3
    assert record_metrics.call_args_list == [
        call(choice="A", score=1.0, completion_index=0),
        call(choice="B", score=0.0, completion_index=1),
        call(choice="A", score=1.0, completion_index=2),
    ]


def test_model_based_classify_rejects_incomplete_multi_sample_result() -> None:
    policy = FakeCompletionFn(["candidate 1", "candidate 2"])
    evaluator = make_model_based_classify(policy, n_samples=3)

    with pytest.raises(ValueError, match="returned 2 completions"):
        evaluator.eval_sample({"input": "question"}, None)
