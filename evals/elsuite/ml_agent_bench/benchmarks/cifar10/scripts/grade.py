import logging
import shutil
from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from torchvision import datasets

import evals.elsuite.ml_agent_bench.benchmarks.cifar10.env.train as baseline_script
from evals.elsuite.ml_agent_bench.low_level_actions import execute_script
from evals.elsuite.ml_agent_bench.utils import get_baseline_score


logger = logging.getLogger(__name__)


def get_score(submission_dir: Path) -> float:
    if not submission_dir.is_dir():
        raise ValueError(f"Expected to be given a directory. Got : {submission_dir}")

    submission_file = submission_dir / "submission.csv"

    if not submission_file.exists():
        return get_naive_baseline_score()  # default to baseline score

    submission = pd.read_csv(submission_file, index_col=0)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True)
    n_correct = 0

    for idx, (x, y_true) in enumerate(test_dataset):
        y_pred = submission.loc[idx].argmax()
        n_correct += int(y_pred == y_true)

    accuracy = 100 * n_correct / len(test_dataset)

    return accuracy


def get_human_baseline_score() -> float:
    """
    Returns the accuracy of the best attempt by a human on the CIFAR-10
    Kaggle leaderboard: https://www.kaggle.com/c/cifar-10/leaderboard.

    Last updated: 2024-01-18.
    """

    return 95.530


@cache
def get_naive_baseline_score() -> float:
    """
    Executes the baseline script `train.py` and returns the accuracy.
    Expects the predictions to be saved to `submission.csv` when run.
    """
    
    scripts_dir = Path(__file__).parent
    env_dir = scripts_dir.parent / "env"
    naive_baseline = env_dir / "train.py"

    return get_baseline_score(naive_baseline, get_score)


def normalize_score(score: float) -> float:
    """
    Transforms the score to be in the range [0, 1], where 0 is the worst
    possible score and 1 is the best possible score.
    """
    return score / 100
