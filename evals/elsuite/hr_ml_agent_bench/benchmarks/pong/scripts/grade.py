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

    if avg_reward < -21.0 or avg_reward > 21.0:
        return get_naive_baseline_score()  # invalid submission

    assert (
        -21.0 <= avg_reward <= 21.0
    ), f"Expected execution time to be non-negative. Got: {avg_reward}"

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

    The minimum reward is -21.0 and the maximum reward is 21.0. See
    https://gymnasium.farama.org/environments/atari/pong/#rewards
    for more information.
    """

    min_score = -21.0
    max_score = 21.0

    return (score - min_score) / (max_score - min_score)
