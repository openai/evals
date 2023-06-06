from typing import Any, Type
from mock import patch
from pytest import mark, raises
from evals.api import DummyCompletionFn
from evals.elsuite.basic.match import Match
from evals.record import DummyRecorder
from evals.utils.test import TestCompletionFn


@mark.parametrize(
    "completion, ideal, expected_match",
    [
        ("world", "world", True),
    ],
)
def test_eval_sample(
    completion: str,
    ideal: list[str],
    expected_match: bool,
):
    eval = Match(
        completion_fns=[TestCompletionFn(completion)],
        samples_jsonl="",
    )

    recorder = DummyRecorder(None)
    with recorder.as_default_recorder("x"), patch.object(
        recorder, "record_match", wraps=recorder.record_match
    ) as record_match:
        eval.eval_sample(dict(input="Hello", ideal=ideal), None)
        record_match.assert_called_once_with(
            expected_match, expected=[ideal], picked=completion, sampled=completion, options=[ideal]
        )


@mark.parametrize(
    "completion, ideal, expected_match",
    [
        ("world", ["world"], True),
    ],
)
def test_eval_sample(
    completion: str,
    ideal: list[str],
    expected_match: bool,
):
    eval = Match(
        completion_fns=[TestCompletionFn(completion)],
        samples_jsonl="",
    )

    recorder = DummyRecorder(None)
    with recorder.as_default_recorder("x"), patch.object(
        recorder, "record_match", wraps=recorder.record_match
    ) as record_match:
        eval.eval_sample(dict(input="Hello", ideal=ideal), None)
        record_match.assert_called_once_with(
            expected_match, expected=ideal, picked=completion, sampled=completion, options=ideal
        )


@mark.parametrize(
    "sample, expected_error",
    [
        (None, AssertionError),
        ("", AssertionError),
        (dict(ideal=42), AssertionError),
        (dict(input="world"), AssertionError),
    ],
)
def test_eval_sample_raises(sample: Any, expected_error: Type):
    eval = Match(
        completion_fns=[DummyCompletionFn()],
        samples_jsonl="",
    )

    with raises(expected_error):
        eval.eval_sample(sample, None)
