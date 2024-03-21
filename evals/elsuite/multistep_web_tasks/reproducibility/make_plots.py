"""This file is supposed to load data from separate task attempts and aggregate it into a usable format"""


import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd

from evals.utils import log_utils

plt.style.use("ggplot")

DESCRIPTIVE_TASK_NAMES = {
    1: "search-wikipedia",
    2: "most-upvoted",
    3: "find-python",
    4: "make-account",
    5: "place-order",
    6: "upvote-posts",
    7: "clone-and-run-repo",
    8: "make-and-push-repo",
    9: "advertise-repo",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    log_dir = args.log_dir
    out_dir = args.out_dir
    df = load_mwt_results_from_dir(log_dir)
    # make_plot(df)
    make_plot(df, out_dir=Path(out_dir))


@dataclass
class MWTTaskOutcome:
    """This class represents the result of one model attempting one task once"""

    solver: str
    task_id: int
    attempt: int
    score: float
    trajectory_length: int


def load_mwt_results_from_dir(log_dir: Union[str, Path]) -> pd.DataFrame:
    task_outcomes = build_task_outcomes(log_dir)
    # apparently you can just build dataframes from lists of dataclasses, that's neat
    df = pd.DataFrame(task_outcomes)
    return df


def build_task_outcomes(log_dir: Union[str, Path]) -> list[MWTTaskOutcome]:
    final_results_dict = log_utils.get_final_results_from_dir(log_dir)
    if any(final_results == "<MISSING RESULTS>" for final_results in final_results_dict.values()):
        print("WARNING: Some results are missing.")
    task_outcomes = []
    for path, final_results in final_results_dict.items():
        if final_results == ("<MISSING RESULTS>"):
            continue
        spec = log_utils.extract_spec(path)
        task_outcome = build_task_outcome(spec, final_results, path)
        task_outcomes.append(task_outcome)
    return task_outcomes


def build_task_outcome(spec: dict, final_results: dict, path: Path) -> MWTTaskOutcome:
    task_id = spec["split"].split("_")[1]
    solver = spec["completion_fns"][0]
    # we have to hackily get the attempt out of the path
    attempt = _get_attempt_number(str(path))
    outcome = MWTTaskOutcome(
        solver=solver,
        task_id=int(task_id),
        attempt=attempt,
        score=final_results["scores"][task_id],
        trajectory_length=final_results["trajectory_lengths"][task_id],
    )
    return outcome


def _get_attempt_number(path: str) -> int:
    # thanks chatgpt:  https://chat.openai.com/share/032bc07f-f676-47a8-a9f0-a46589ca4281
    pattern = r"attempt_(\d+)"
    match = re.search(pattern, path)

    if match:
        attempt_number = match.group(1)
        return int(attempt_number)
    else:
        raise ValueError(f"Could not find attempt number in {path}")


def make_plot(df: pd.DataFrame, out_dir: Path) -> None:
    # thanks chatgpt: https://chat.openai.com/share/3e9b1957-7941-4121-a40c-2fa9f6a9b371

    # Rename task_id to use descriptive names
    names_to_replace = {i: f"{i}_{DESCRIPTIVE_TASK_NAMES[i]}" for i in DESCRIPTIVE_TASK_NAMES}
    df["task_id"] = df["task_id"].replace(names_to_replace)

    # Group by task_id and solver
    grouped = df.groupby(["task_id", "solver"])

    # Calculate the fraction of attempts with score 1 for each group
    fractions = grouped["score"].mean().reset_index()

    # Pivot the data for plotting
    pivot = fractions.pivot(index="task_id", columns="solver", values="score")

    # Plot the data
    ax = pivot.plot(kind="bar", figsize=(10, 5))

    # Set the labels and title
    ax.set_ylabel("Fraction of Attempts Successful")
    ax.set_xlabel("Task")
    ax.set_title("Fraction of Successful Attempts for Each Task and Solver")

    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=-45, ha="left")

    # Show the legend
    labels = [
        "strong/gpt-3.5\n-turbo-16k-0613",
        "strong/gpt-4\n-32k-0613",
    ]
    ax.legend(labels=labels, title="Solver Type", loc="center left", bbox_to_anchor=(1, 0.5))

    out_dir.mkdir(parents=True)
    plt.tight_layout()
    plt.savefig(out_dir / "fraction-successful-attempts.png")


if __name__ == "__main__":
    main()
