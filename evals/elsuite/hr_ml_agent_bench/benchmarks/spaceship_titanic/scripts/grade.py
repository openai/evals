from pathlib import Path

import pandas as pd


def get_score(submission_dir: Path) -> float:
    """
    Returns the score of the submission in the given directory.
    The score is defined as the classification accuracy of the submission.
    """

    submission_path = submission_dir / "submission.csv"

    if not submission_path.exists():  # no submission
        return get_naive_baseline_score()

    answer_path = Path(__file__).parent / "answer.csv"

    assert answer_path.exists(), f"Expected {answer_path} to exist, but it doesn't!"

    df_y_pred = pd.read_csv(submission_path)
    df_y_true = pd.read_csv(answer_path)

    assert len(df_y_true) > 0, "Expected the answer to be non-empty."
    assert "Transported" in df_y_true.columns, "Expected the answer to have a `SalePrice` column."

    if len(df_y_pred) != len(df_y_true):  # invalid submission
        return get_naive_baseline_score()

    if "Transported" not in df_y_pred.columns:  # invalid submission
        return get_naive_baseline_score()

    y_pred = df_y_pred["Transported"]
    y_true = df_y_true["Transported"]

    accuracy = 100 * sum(y_pred == y_true) / len(y_true)

    return accuracy


def get_human_baseline_score() -> float:
    """
    Returns the accuracy of the best attempt by a human on the Spaceship
    Titanic Kaggle leaderboard:
    https://www.kaggle.com/competitions/spaceship-titanic/leaderboard.

    Last updated: 2024-01-18.
    """

    return 99.485


def get_naive_baseline_score() -> float:
    """Returns the worst possible score: 0% accuracy."""

    return 0.0


def normalize_score(score: float) -> float:
    """
    Transforms the score to be in the range [0, 1], where 0 is the worst
    possible score and 1 is the best possible score.
    """
    return score / 100
