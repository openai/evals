from evals.elsuite.modelgraded.classify_utils import INVALID_STR, get_choice_score


def test_invalid_choice_has_no_numeric_score() -> None:
    assert get_choice_score(INVALID_STR, ["A", "B"], {"A": 1.0, "B": 0.0}) is None


def test_invalid_choice_has_no_score_for_from_strings() -> None:
    assert get_choice_score(INVALID_STR, ["1", "2"], "from_strings") is None


def test_valid_choice_keeps_configured_score() -> None:
    assert get_choice_score("A", ["A", "B"], {"A": 1.0, "B": 0.0}) == 1.0
