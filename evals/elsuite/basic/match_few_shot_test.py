from pathlib import Path

from mock import patch
from pytest import raises

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
