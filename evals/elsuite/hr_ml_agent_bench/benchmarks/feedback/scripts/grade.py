from pathlib import Path

import numpy as np
import pandas as pd

from evals.elsuite.hr_ml_agent_bench.benchmarks.feedback.env.train import (
    DIMENSIONS,
    compute_metrics_for_regression,
)


def get_score(submission_dir: Path) -> float:
    submission_path = submission_dir / "submission.csv"

    if not submission_path.exists():  # no submission
        return get_naive_baseline_score()

    answer_path = Path(__file__).parent / "answer.csv"

    assert answer_path.exists(), f"Expected {answer_path} to exist, but it doesn't!"

    df_y_pred = pd.read_csv(submission_path)
    df_y_true = pd.read_csv(answer_path)

    assert len(df_y_true) > 0, "Expected the answer to be non-empty."

    for dim in DIMENSIONS:
        assert (
            dim in df_y_true.columns
        ), f"Expected the answer to have a `{dim}` column. Got: {df_y_true.columns}."

    if len(df_y_pred) != len(df_y_true):  # invalid submission
        return get_naive_baseline_score()

    for dim in DIMENSIONS:
        if dim not in df_y_pred.columns:  # invalid submission
            return get_naive_baseline_score()

    y_pred = df_y_pred[DIMENSIONS].to_numpy()
    y_true = df_y_true[DIMENSIONS].to_numpy()

    errors = compute_metrics_for_regression(y_test=y_true, y_test_pred=y_pred)
    avg_error = np.mean(list(errors.values()))
    return avg_error


def get_human_baseline_score() -> float:
    """
    Returns the "Multi-class logarithmic loss" of the best attempt by a human
    on the Feedback Prize Kaggle leaderboard:
    https://www.kaggle.com/competitions/feedback-prize-effectiveness/leaderboard.

    Last updated: 2024-01-18.
    """

    return 0.55435


def get_naive_baseline_score() -> float:
    """
    Returns the "Multi-class logarithmic loss" of the weakest attempt by a
    human on the Feedback Prize Kaggle leaderboard:
    https://www.kaggle.com/competitions/feedback-prize-effectiveness/leaderboard.

    Last updated: 2024-01-24.
    """

    return 18.44720


def normalize_score(score: float) -> float:
    """
    Transforms the score to be in the range [0, 1], where 0 is the worst
    possible score and 1 is the best possible score.
    """
    # Remap [inf, 0] to [0, 1]
    return 1 / (1 + score)
