from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error


def get_score(submission_dir: Path) -> float:
    """Returns the score of the submission in the given directory."""

    submission_path = submission_dir / "submission.csv"

    if not submission_path.exists():  # no submission
        return get_naive_baseline_score()

    answer_path = Path(__file__).parent / "answer.csv"

    assert answer_path.exists(), f"Expected {answer_path} to exist, but it doesn't!"

    df_y_pred = pd.read_csv(submission_path)
    df_y_true = pd.read_csv(answer_path)

    assert len(df_y_true) > 0, "Expected the answer to be non-empty."
    assert "SalePrice" in df_y_true.columns, "Expected the answer to have a `SalePrice` column."

    if len(df_y_pred) != len(df_y_true):  # invalid submission
        return get_naive_baseline_score()

    if "SalePrice" not in df_y_pred.columns:  # invalid submission
        return get_naive_baseline_score()

    y_pred = df_y_pred["SalePrice"]
    y_true = df_y_true["SalePrice"]

    assert (y_true > 0).all(), f"Expected `y_true` to contain only positive values. Got: {y_true}."

    if (y_pred <= 0).any():  # invalid submission
        return get_naive_baseline_score()

    log_y_pred = np.log(y_pred)
    log_y_true = np.log(y_true)

    rmse = root_mean_squared_error(y_true=log_y_true, y_pred=log_y_pred)

    return rmse


def get_human_baseline_score() -> float:
    """
    Returns the RMSE of the best attempt by a human on the House Prices
    Kaggle leaderboard:
    https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/leaderboard.

    Last updated: 2024-01-18.
    """

    return 0.0


def get_naive_baseline_score() -> float:
    """
    Returns the RMSE of the weakest attempt by a human on the House Prices
    Kaggle leaderboard:
    https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/leaderboard.

    Last updated: 2024-01-24.
    """

    return 31.42506


def normalize_score(score: float) -> float:
    """
    Transforms the score to be in the range [0, 1], where 0 is the worst
    possible score and 1 is the best possible score.
    """
    # Remap [inf, 0] to [0, 1]
    return 1 / (1 + score)
