from pathlib import Path
from typing import Any, Type

from mock import patch
from pytest import mark, raises

from evals.api import DummyCompletionFn
from evals.elsuite.basic.json_match import JsonMatch
from evals.record import DummyRecorder
from evals.utils.test import TestCompletionFn


@mark.parametrize(
    "completion, ideal, expected_metrics",
    [
        # Basic match
        ('{ "key": "value" }', '{ "key": "value" }', dict(accuracy=1.0)),
        # Whitespace is not significant
        ('{\n   "key":"value"\n   }\n', '{ "key": "value" }', dict(accuracy=1.0)),
        # Key order is not significant
        (
            '{ "key2": "foo", "key1": "bar" }',
            '{ "key1": "bar", "key2": "foo" }',
            dict(accuracy=1.0),
        ),
        # No match if values are different
        ('{ "key": "value" }', '{ "key": "notvalue" }', dict(accuracy=0)),
        # Values can be numbers as well as strings
        ('{ "key": 100 }', '{ "key": 100 }', dict(accuracy=1.0)),
        # Numerical values are not accepted if they differ
        ('{ "key": 100 }', '{ "key": 100.1 }', dict(accuracy=0)),
        # Completion is accepted if it is found in an array of valid answers
        ('{ "key": 100 }', ['{ "key": 100.1 }', '{ "key": 100 }'], dict(accuracy=1.0)),
        # Completion is not accepted if it is not found in an array of valid answers
        ('{ "key": 100 }', ['{ "key": 100.1 }', '{ "key": 99.9 }'], dict(accuracy=0)),
        # Different keys do not match
        ('{ "key": "value" }', '{ "anotherkey": "value" }', dict(accuracy=0)),
        # Missing keys do not match
        (
            '{ "key": "value" }',
            '{ "key": "value", "anotherkey": "value" }',
            dict(accuracy=0),
        ),
        # Extra keys do not match
        (
            '{ "key": "value", "anotherkey": "value" }',
            '{ "key": "value" }',
            dict(accuracy=0),
        ),
        # Lists are supported, and matched by element equality
        ('{ "key": [1.0,2.0,3.0] }', '{ "key": [1, 2, 3] }', dict(accuracy=1.0)),
        # Lists of different lengths do not match
        ('{ "key": [1, 2, 3] }', '{ "key": [1, 2, 3, 3] }', dict(accuracy=0)),
        # Lists that are not equal index-by-index do not match
        ('{ "key": [1, 2, 3] }', '{ "key": [1, 3, 2] }', dict(accuracy=0)),
        # An empty list does not match a nonempty list
        ('{ "key": [] }', '{ "key": [1] }', dict(accuracy=0)),
        # Completion with invalid JSON is not accepted
        ('{ "key": "value }', '{ "key": "value" }', dict(accuracy=0)),
    ],
)
def test_eval_sample(
    completion: str,
    ideal: list[str],
    expected_metrics: dict[str, float],
) -> None:
    eval = JsonMatch(
        completion_fns=[TestCompletionFn(completion)],
        samples_jsonl="",
        eval_registry_path=Path("."),
    )

    recorder = DummyRecorder(None)
    with recorder.as_default_recorder("x"), patch.object(
        recorder, "record_metrics", wraps=recorder.record_metrics
    ) as record_metrics:
        eval.eval_sample(dict(input=completion, ideal=ideal), None)
        record_metrics.assert_called_once_with(**expected_metrics)


@mark.parametrize(
    "sample, expected_error",
    [
        (None, AssertionError),
        ("", AssertionError),
        (dict(ideal="world"), AssertionError),  # Missing input
        (dict(input="world"), AssertionError),  # Missing ideal answer
    ],
)
def test_eval_sample_raises(sample: Any, expected_error: Type[Exception]) -> None:
    eval = JsonMatch(
        completion_fns=[DummyCompletionFn()],
        samples_jsonl="",
        eval_registry_path=Path("."),
    )

    with raises(expected_error):
        eval.eval_sample(sample, None)
