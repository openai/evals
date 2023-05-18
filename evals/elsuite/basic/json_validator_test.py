from typing import Any, Type
from mock import patch
from pytest import mark, raises
from evals.api import DummyCompletionFn
from evals.elsuite.basic.json_validator import JsonValidator
from evals.record import DummyRecorder
from evals.utils.test import TestCompletionFn


@mark.parametrize(
    "completion, expected_match",
    [
        ('{"foo": "bar"}', True),
        ('notjson', False),
    ],
)
def test_eval_sample(
    completion: str,
    expected_match: bool,
):
    eval = JsonValidator(
        completion_fns=[TestCompletionFn(completion)],
        samples_jsonl="",
    )

    recorder = DummyRecorder(None)
    with recorder.as_default_recorder("x"), patch.object(
        recorder, "record_match", wraps=recorder.record_match
    ) as record_match:
        eval.eval_sample(dict(input="Hello"), None)
        record_match.assert_called_once_with(
            expected_match, expected=None, picked=completion
        )


@mark.parametrize(
    "sample, expected_error",
    [
        (None, AssertionError),
        ("", AssertionError),
        ({}, AssertionError),
    ],
)
def test_eval_sample_raises(sample: Any, expected_error: Type):
    eval = JsonValidator(
        completion_fns=[DummyCompletionFn()],
        samples_jsonl="",
    )

    with raises(expected_error):
        eval.eval_sample(sample, None)
