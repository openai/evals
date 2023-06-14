from typing import Any, Type
from mock import patch
from pytest import mark, raises
from evals.api import DummyCompletionFn
from evals.elsuite.basic.fuzzy_match import FuzzyMatch
from evals.record import DummyRecorder
from evals.utils.test import TestCompletionFn


@mark.parametrize(
    "completion, ideal, expected_metrics",
    [
        ("world", "world", dict(accuracy=1.0, f1_score=1.0)),
        ("world", "foo", dict(accuracy=0, f1_score=0)),
        ("world", ["some foo world", "dummy"], dict(accuracy=1.0, f1_score=0.5)),
    ],
)
def test_eval_sample(
    completion: str,
    ideal: list[str],
    expected_metrics: dict[str, float],
):
    eval = FuzzyMatch(
        completion_fns=[TestCompletionFn(completion)],
        samples_jsonl="",
    )

    recorder = DummyRecorder(None)
    with recorder.as_default_recorder("x"), patch.object(
        recorder, "record_metrics", wraps=recorder.record_metrics
    ) as record_metrics:
        eval.eval_sample(dict(input="Hello", ideal=ideal), None)
        record_metrics.assert_called_once_with(**expected_metrics)


@mark.parametrize(
    "sample, expected_error",
    [
        (None, AssertionError),
        ("", AssertionError),
        (dict(ideal="world"), AssertionError),
        (dict(input="world"), AssertionError),
    ],
)
def test_eval_sample_raises(sample: Any, expected_error: Type):
    eval = FuzzyMatch(
        completion_fns=[DummyCompletionFn()],
        samples_jsonl="",
    )

    with raises(expected_error):
        eval.eval_sample(sample, None)
