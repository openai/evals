from functools import cache
from pathlib import Path

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

    return avg_reward


@cache
def get_human_baseline_score() -> float:
    """
    Executes the human baseline and returns its score.
    """

    scripts_dir = Path(__file__).parent
    baselines_dir = scripts_dir.parent / "baselines"
    human_baseline = baselines_dir / "human.py"

    return get_baseline_score(human_baseline, get_score)


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

    The possible total return for a `BipedalWalker-v3` episode is in the
    range [-279.3, 330]. The minimum is achieved by the walker squatting
    in-place and hitting the ground in the final frame, thereby maximising
    the control penalty and receving a -100 reward for falling. The maximum
    reward is achieved by the walker moving forward at maximum speed (it
    was emperically observed to never exceed 330.0 in practice, but the
    theoretical maximum is unknown).

    See https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/box2d/bipedal_walker.py
    for more information about the reward function.
    """

    min_score = -279.3
    max_score = 330.0

    return (score - min_score) / (max_score - min_score)
