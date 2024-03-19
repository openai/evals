from pathlib import Path

import pandas as pd
from datasets import load_dataset


def get_score(submission_dir: Path) -> float:
    if not submission_dir.is_dir():
        raise ValueError(f"Expected to be given a directory. Got: {submission_dir}")

    submission_file = submission_dir / "submission.csv"

    if not submission_file.exists():
        return get_naive_baseline_score()  # default to baseline score

    submission = pd.read_csv(submission_file, index_col="idx")
    imdb = load_dataset("imdb")
    n_correct = 0

    for idx, data in enumerate(imdb["test"]):
        y_true = data["label"]

        try:
            y_pred = submission.loc[idx].argmax()
        except KeyError:  # invalid submission
            return get_naive_baseline_score()

        n_correct += int(y_pred == y_true)

    accuracy = 100 * n_correct / len(imdb["test"])

    return accuracy


def get_human_baseline_score() -> float:
    """
    Returns the accuracy of the best attempt by a human
    on the IMDb Papers with Code leaderboard:
    https://paperswithcode.com/sota/sentiment-analysis-on-imdb.

    Last updated: 2024-01-18.
    """

    return 96.21


def get_naive_baseline_score() -> float:
    """
    The baseline script is only partially implemented and therefore
    doesn't produce a submission file. The baseline score
    defaults to 0% accuracy.
    """

    return 0.0


def normalize_score(score: float) -> float:
    """
    Transforms the score to be in the range [0, 1], where 0 is the worst
    possible score and 1 is the best possible score.
    """
    return score / 100
