from pathlib import Path
from typing import Any, Type, Union

from mock import patch
from pytest import mark, raises

from evals.api import DummyCompletionFn
from evals.elsuite.basic.includes import Includes
from evals.record import DummyRecorder
from evals.utils.test import TestCompletionFn


@mark.parametrize(
    "completion, ideal, expected_match, ignore_case, use_exclusions, excluded_terms",
    [
        ("world", "world", True, False, False, None),
        ("world", "wOrLd", True, True, False, None),
        ("world", ["world"], True, False, False, None),
        ("world", ["foo", "bar"], False, False, False, None),
        ("world", ["worldfoo", "worldbar"], False, False, False, None),
        # test for exclusions: does including an excludable word lead to False match, on what would otherwise be True match?
        ("world exclusion", "world", False, False, True, ["exclusion", "excluded"]),
        # change the 2nd word in completion from an excluded word, to make sure this would be true otherwise
        ("world okay", "world", True, False, True, ["exclusion", "excluded"]),
    ],
)
def test_eval_sample(
    completion: str,
    ideal: Union[str, list[str]],
    expected_match: bool,
    ignore_case: bool,
    use_exclusions: bool = False,
    excluded_terms: Union[str, list[str]] = []
):
    eval = Includes(
        completion_fns=[TestCompletionFn(completion)],
        samples_jsonl="",
        eval_registry_path=Path("."),
        ignore_case=ignore_case,
        use_exclusions=use_exclusions,
        excluded_terms=excluded_terms,
    )

    recorder = DummyRecorder(None)
    with recorder.as_default_recorder("x"), patch.object(
        recorder, "record_match", wraps=recorder.record_match
    ) as record_match:
        sample_dict = dict(input="Hello", ideal=ideal, exclude=excluded_terms)
        eval.eval_sample(sample_dict, None)
        record_match.assert_called_once_with(
            expected_match, expected=ideal, picked=completion, sampled=completion,
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
        eval_registry_path=Path("."),
    )

    with raises(expected_error):
        eval.eval_sample(sample, None)
