from evals.elsuite.modelgraded.classify_utils import INVALID_STR, get_choice_score


def test_invalid_choice_has_no_numeric_score() -> None:
    assert get_choice_score(INVALID_STR, ["A", "B"], {"A": 1.0, "B": 0.0}) is None


def test_invalid_choice_has_no_score_for_from_strings() -> None:
    assert get_choice_score(INVALID_STR, ["1", "2"], "from_strings") is None


def test_valid_choice_keeps_configured_score() -> None:
    assert get_choice_score("A", ["A", "B"], {"A": 1.0, "B": 0.0}) == 1.0


def test_invalid_outputs_remain_counted_but_do_not_lower_mean_score():
    from unittest.mock import Mock

    from evals.elsuite.modelgraded.classify import ModelBasedClassify

    evaluator = ModelBasedClassify.__new__(ModelBasedClassify)
    evaluator.get_samples = Mock(return_value=[])
    evaluator.eval_all_samples = Mock()
    recorder = Mock()
    recorder.get_metrics.return_value = [
        {"choice": "A", "score": get_choice_score("A", ["A", "B"], {"A": 1.0, "B": 0.0})},
        {
            "choice": INVALID_STR,
            "score": get_choice_score(INVALID_STR, ["A", "B"], {"A": 1.0, "B": 0.0}),
        },
    ]
    assert evaluator.run(recorder) == {"counts/A": 1, f"counts/{INVALID_STR}": 1, "score": 1.0}
    recorder.get_metrics.return_value = [{"choice": INVALID_STR, "score": None}]
    assert evaluator.run(recorder) == {f"counts/{INVALID_STR}": 1}
