from unittest.mock import patch

from evals.elsuite.basic.exact_match import record_and_check_exact_match


def test_exact_match_accepts_identical_response() -> None:
    with patch("evals.elsuite.basic.exact_match.record_match") as record_match:
        picked = record_and_check_exact_match("42", "42")

    assert picked == "42"
    record_match.assert_called_once_with(
        True,
        expected=["42"],
        picked="42",
        sampled="42",
        options=["42"],
    )


def test_exact_match_rejects_prefix_that_match_accepts() -> None:
    with patch("evals.elsuite.basic.exact_match.record_match") as record_match:
        picked = record_and_check_exact_match("42 extra", "42")

    assert picked is None
    record_match.assert_called_once_with(
        False,
        expected=["42"],
        picked=None,
        sampled="42 extra",
        options=["42"],
    )


def test_exact_match_accepts_any_identical_reference() -> None:
    with patch("evals.elsuite.basic.exact_match.record_match") as record_match:
        picked = record_and_check_exact_match("four", ["4", "four"])

    assert picked == "four"
    record_match.assert_called_once_with(
        True,
        expected=["4", "four"],
        picked="four",
        sampled="four",
        options=["4", "four"],
    )


def test_exact_match_is_whitespace_sensitive() -> None:
    with patch("evals.elsuite.basic.exact_match.record_match") as record_match:
        picked = record_and_check_exact_match(" 42", "42")

    assert picked is None
    record_match.assert_called_once_with(
        False,
        expected=["42"],
        picked=None,
        sampled=" 42",
        options=["42"],
    )
