from typing import Any, Type, Union
from mock import patch
from pytest import mark, raises
from evals.api import DummyCompletionFn
from evals.elsuite.basic.includes import Includes
from evals.record import DummyRecorder
from evals.utils.test import TestCompletionFn


@mark.parametrize(
    "completion, ideal, expected_match, ignore_case",
    [
        ("world", "world", True, False),
        ("world", "wOrLd", True, True),
        ("world", ["world"], True, False),
        ("world", ["foo", "bar"], False, False),
        ("world", ["worldfoo", "worldbar"], False, False),
    ],
)
def test_eval_sample(
    completion: str,
    ideal: Union[str, list[str]],
    expected_match: bool,
    ignore_case: bool,
):
    eval = Includes(
        completion_fns=[TestCompletionFn(completion)],
        samples_jsonl="",
        ignore_case=ignore_case,
    )

    recorder = DummyRecorder(None)
    with recorder.as_default_recorder("x"), patch.object(
        recorder, "record_match", wraps=recorder.record_match
    ) as record_match:
        eval.eval_sample(dict(input="Hello", ideal=ideal), None)
        record_match.assert_called_once_with(
            expected_match, expected=ideal, picked=completion, sampled=completion
        )


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
    eval = Includes(
        completion_fns=[DummyCompletionFn()],
        samples_jsonl="",
    )

    with raises(expected_error):
        eval.eval_sample(sample, None)
