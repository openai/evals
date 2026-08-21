from unittest.mock import patch

from evals.api import record_and_check_match


def test_record_and_check_match_can_ignore_case() -> None:
    with patch("evals.api.record_match") as record_match:
        picked = record_and_check_match(
            prompt="prompt",
            sampled="No",
            expected="no",
            ignore_case=True,
        )

    assert picked == "no"
    record_match.assert_called_once_with(
        True,
        expected=["no"],
        picked="no",
        sampled="No",
        options=["no"],
    )


def test_record_and_check_match_can_strip_whitespace() -> None:
    with patch("evals.api.record_match") as record_match:
        picked = record_and_check_match(
            prompt="prompt",
            sampled="  Mumbai\n",
            expected="Mumbai",
            strip=True,
        )

    assert picked == "Mumbai"
    record_match.assert_called_once_with(
        True,
        expected=["Mumbai"],
        picked="Mumbai",
        sampled="  Mumbai\n",
        options=["Mumbai"],
    )


def test_record_and_check_match_defaults_remain_strict() -> None:
    with patch("evals.api.record_match") as record_match:
        case_mismatch = record_and_check_match(
            prompt="prompt",
            sampled="No",
            expected="no",
        )
        whitespace_mismatch = record_and_check_match(
            prompt="prompt",
            sampled=" Mumbai",
            expected="Mumbai",
        )

    assert case_mismatch is None
    assert whitespace_mismatch is None
    assert record_match.call_count == 2
