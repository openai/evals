from pathlib import Path

from mock import patch
from pytest import mark, raises

from evals.api import DummyCompletionFn
from evals.elsuite.basic.match import Match


def _make_match() -> Match:
    return Match(
        completion_fns=[DummyCompletionFn()],
        samples_jsonl="samples.jsonl",
        num_few_shot=1,
        few_shot_jsonl="few-shot.jsonl",
        eval_registry_path=Path("."),
    )


def test_missing_sample_key_reports_dataset_and_row() -> None:
    rows = [{"input": "example"}]
    with patch("evals.elsuite.basic.match.evals.get_jsonl", return_value=rows):
        with raises(ValueError, match=r"Few-shot row 0.*few-shot.jsonl.*sample"):
            _make_match()


def test_non_chat_sample_is_rejected_during_initialization() -> None:
    rows = [{"sample": "example"}]
    with patch("evals.elsuite.basic.match.evals.get_jsonl", return_value=rows):
        with raises(ValueError, match=r"Few-shot row 0.*invalid.*sample"):
            _make_match()


@mark.parametrize(
    "message",
    [
        {},
        {"role": "user"},
        {"content": "hello"},
        {"role": 7, "content": "hello"},
        {"role": "user", "content": None},
        {"role": "", "content": "hello"},
    ],
)
def test_malformed_chat_messages_are_rejected_before_evaluation(message):
    with patch("evals.elsuite.basic.match.evals.get_jsonl", return_value=[{"sample": [message]}]):
        with raises(ValueError, match=r"Few-shot row 0.*message 0"):
            _make_match()


def test_valid_chat_messages_preserve_optional_fields():
    rows = [
        {
            "sample": [
                {"role": "user", "content": "hello", "name": "example_user"},
                {"role": "assistant", "content": "answer"},
            ]
        }
    ]
    with patch("evals.elsuite.basic.match.evals.get_jsonl", return_value=rows):
        evaluator = _make_match()
    assert evaluator.few_shot == rows
