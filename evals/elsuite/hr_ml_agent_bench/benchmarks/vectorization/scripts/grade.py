import re
from functools import cache
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import evals.elsuite.hr_ml_agent_bench.benchmarks.vectorization.env.train as baseline_script
from evals.elsuite.hr_ml_agent_bench.low_level_actions import execute_script


def get_score(submission_dir: Path) -> float:
    if not submission_dir.is_dir():
        raise ValueError(f"Expected to be given a directory. Got: {submission_dir}")

    submission_file = submission_dir / "submission.txt"

    if not submission_file.exists():
        return get_naive_baseline_score()  # no submission

    with open(submission_file, "r") as f:
        try:
            execution_time = float(f.read())
        except ValueError:
            return get_naive_baseline_score()  # invalid submission

    if execution_time < 0:
        return get_naive_baseline_score()  # invalid submission

    assert execution_time >= 0, f"Expected execution time to be non-negative. Got: {execution_time}"
    return execution_time


def _get_execution_time(scriptpath: Path) -> float:
    with TemporaryDirectory() as tmp_dir:
        tmp_baseline_fpath = Path(tmp_dir) / scriptpath.name

        copyfile(
            src=scriptpath,
            dst=tmp_baseline_fpath,
        )

        output = execute_script(
            script_name=tmp_baseline_fpath,
            device=0,
            python="python",
            work_dir=tmp_baseline_fpath.parent,
        )

    pattern = r"Time taken for execution: (\d+(\.\d+)?) seconds"
    match = re.search(pattern, output)

    try:
        execution_time = float(match.group(1))
    except AttributeError:
        raise RuntimeError(
            f"Could not find score in script output of {scriptpath}! "
            "Expected baseline script to print score in the following format: "
            "'Time taken for execution: \{interval_time\} seconds'."
        )
    except ValueError:
        raise RuntimeError(
            f"Could not convert score to float! Got: {match.group(1)}" " but expected a float."
        )

    assert (
        execution_time >= 0
    ), f"Expected execution time to be non-negative. Got: {execution_time}."
    return execution_time


def get_human_baseline_score() -> float:
    """
    Executes human baseline script `human_baseline.py` and returns the
    execution time. Expects the score to be printed to stdout and to follow the
    following pattern: 'Time taken for execution: {interval_time} seconds'.
    """

    baseline_fpath = Path(__file__).parent / "human_baseline.py"
    assert baseline_fpath.exists(), f"Couldn't find human baseline script at {baseline_fpath}!"
    return _get_execution_time(baseline_fpath)


@cache
def get_naive_baseline_score() -> float:
    """
    Executes naive baseline script `train.py` and returns the execution time.
    Expects the score to be printed to stdout and to follow the following
    pattern: 'Time taken for execution: {interval_time} seconds'.
    """

    baseline_fpath = Path(baseline_script.__file__)
    assert baseline_fpath.exists(), f"Couldn't find naive baseline script at {baseline_fpath}!"
    return _get_execution_time(baseline_fpath)


def normalize_score(score: float) -> float:
    """
    Transforms the score to be in the range [0, 1], where 0 is the worst
    possible score and 1 is the best possible score.
    """
    return 1 / (1 + score)  # Map [inf, 0] -> [0, 1]
