import logging
import shutil
from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset

from evals.elsuite.hr_ml_agent_bench.utils import get_root_dir

logger = logging.getLogger(__name__)


def get_score(submission_dir: Path) -> float:
    if not submission_dir.is_dir():
        raise ValueError(f"Expected to be given a directory. Got: {submission_dir}")

    submission_file = submission_dir / "submission.csv"

    if not submission_file.exists():
        return get_naive_baseline_score()  # default to baseline score

    submission = pd.read_csv(submission_file)
    target_dataset = "ogbn-arxiv"

    dataset = PygNodePropPredDataset(name=target_dataset, root="networks")
    data = dataset[0]
    split_idx = dataset.get_idx_split()

    test_idx = split_idx["test"]

    evaluator = Evaluator(name=target_dataset)
    y_true = data.y.cpu()

    submission = torch.tensor(np.array(submission))

    test_acc = evaluator.eval(
        {
            "y_true": y_true[test_idx],
            "y_pred": submission,
        }
    )["acc"]

    return test_acc


def get_human_baseline_score() -> float:
    """
    Returns the test accuracy of the best attempt by a human on the
    ogbn-arxiv leaderboard:
    https://ogb.stanford.edu/docs/leader_nodeprop/#ogbn-arxiv.

    Last updated: 2024-01-18.
    """

    return 78.03


@cache
def get_naive_baseline_score() -> float:
    """
    Executes the baseline script `train.py` and returns the accuracy.
    Assumes the predictions are saved to `submission.csv` when
    `train.py` is run.
    """

    env_dir = Path(__file__).parent / ".." / "env"
    dataset_dir = get_root_dir() / "registry" / "data" / "hr_ml_agent_bench" / "ogbn_arxiv" / "dataset"

    with TemporaryDirectory() as tmp_dir:
        dst_dir = Path(tmp_dir) / "env"

        shutil.copytree(
            src=env_dir / "networks",
            dst=dst_dir / "networks",
        )

        shutil.copyfile(
            src=dataset_dir / "baseline.csv",
            dst=dst_dir / "submission.csv",
        )

        accuracy = get_score(dst_dir)

    return accuracy


def normalize_score(score: float) -> float:
    """
    Transforms the score to be in the range [0, 1], where 0 is the worst
    possible score and 1 is the best possible score.
    """
    # Identity; already normalized
    return score
