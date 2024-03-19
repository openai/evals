from functools import cache
from pathlib import Path

from evals.elsuite.hr_ml_agent_bench.utils import get_baseline_score


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

    if avg_reward < 0:
        return get_naive_baseline_score()  # invalid submission

    assert avg_reward >= 0, f"Expected execution time to be non-negative. Got: {avg_reward}"

    return avg_reward


def get_human_baseline_score() -> float:
    """
    Executes the human baseline and returns its score.
    """

    return 1_000.0


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

    The maximum score for the InvertedPendulum-v4 environment is 1,000 (
    https://gymnasium.farama.org/environments/mujoco/inverted_pendulum/#rewards),
    so we divide the score by 1,000 to normalize to the range [0, 1].
    """

    return score / 1_000.0
