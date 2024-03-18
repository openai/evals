from pathlib import Path

import pandas as pd

from evals.elsuite.ml_agent_bench.benchmarks.parkinsons_disease.env.train import smapep1


def get_score(submission_dir: Path) -> float:
    """
    Returns 100 - % error, where '% error' is the symmetric
    mean absolute percentage error (SMAPE) of the
    predictions stored in `submission.csv`. Higer is better.
    """

    submission_fname = submission_dir / "submission.csv"
    answers_fname = Path(__file__).parent / "answer.csv"

    if not submission_fname.exists():
        return get_naive_baseline_score()  # default to baseline score

    assert answers_fname.exists(), f"Expected {answers_fname.as_posixs()} to exist, but it doesn't."

    y_true = pd.read_csv(answers_fname)
    y_pred = pd.read_csv(submission_fname)

    percentage_error = smapep1(
        y_true=y_true["rating"],
        y_pred=y_pred["rating"],
    )
    return percentage_error


def get_human_baseline_score() -> float:
    """
    Returns the % error of the best attempt by a human on the
    AMP-Parkinson's Disease Progression Prediction Kaggle
    leaderboard:
    https://www.kaggle.com/competitions/amp-parkinsons-disease-progression-prediction/leaderboard.

    Last updated: 2024-01-18.
    """

    return 60.042


def get_naive_baseline_score() -> float:
    """Returns the worst possible score: a 100% error rate."""

    return 100.0


def normalize_score(score: float) -> float:
    """
    Transforms the score to be in the range [0, 1], where 0 is the worst
    possible score and 1 is the best possible score.
    """
    # Map [100, 0] to [0, 1]
    return (100 - score) / 100
