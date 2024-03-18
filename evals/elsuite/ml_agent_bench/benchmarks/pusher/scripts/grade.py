from functools import cache
from pathlib import Path

import numpy as np

from evals.elsuite.ml_agent_bench.utils import get_baseline_score


def get_score(submission_dir: Path) -> float:
    """
    Returns the score (the average reward over 100 consecutive
    episodes) of the submission in the given directory.
    """

    if not submission_dir.exists():
        raise ValueError(f"Expected to be given a directory. Got: {submission_dir}")

    submission_file = submission_dir / "submission.txt"

    if not submission_file.exists():
        return get_naive_baseline_score()  # no submission

    with open(submission_file, "r") as f:
        try:
            avg_reward = float(f.read())
        except ValueError:
            return get_naive_baseline_score()  # invalid submission

    if avg_reward > 0:
        return get_naive_baseline_score()  # invalid submission

    assert avg_reward <= 0, f"Expected execution time to be non-positive. Got: {avg_reward}"

    return avg_reward


@cache
def get_human_baseline_score() -> float:
    """
    Executes the human baseline and returns its score.
    """

    scripts_dir = Path(__file__).parent
    baselines_dir = scripts_dir.parent / "baselines"
    human_baseline = baselines_dir / "human.py"
    checkpoint = baselines_dir / "human.checkpoint"

    other_files = []

    if checkpoint.exists():
        other_files = [checkpoint]

    return get_baseline_score(
        baseline_script=human_baseline,
        score_fn=get_score,
        other_files=other_files,
    )


@cache
def get_naive_baseline_score() -> float:
    """
    Executes the naive baseline and returns its score.
    """

    scripts_dir = Path(__file__).parent
    baselines_dir = scripts_dir.parent / "baselines"
    naive_baseline = baselines_dir / "naive.py"

    return get_baseline_score(naive_baseline, get_score)


def normalize_score(score: float) -> float:
    """
    Transforms the score to be in the range [0, 1], where 0 is the worst
    possible score and 1 is the best possible score.
    """

    # The minimum reward (-inf) is achieved by pushing the puck infinitely
    # far from the target. However, a naive agent scores roughly -150 on
    # average, which we take as the empirical minimum. See
    # https://gymnasium.farama.org/environments/mujoco/pusher for more
    # details about the environment.
    min_score = -150.0

    # The maximum reward (0) is achieved by pushing the puck to the target.
    max_score = 0.0

    # Use a linear transformation to avoid an unintuitive warping of the score.
    clipped = np.clip(score, min_score, max_score)
    score = (clipped - min_score) / (max_score - min_score)

    return score
