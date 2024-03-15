import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

TASK_PREFIX = {
    "dyck_languages": (
        "Given the following sequence of opening and closing brackets, "
        "provide the minimal sequence of additional brackets that would "
        "balance the original sequence:\n"
    ),
    "logical_deduction": "",
    "multistep_arithmetic": "",
    "tracking_shuffled_objects": "",
    "word_sorting": "Sort the following list of words alphabetically:\n",
}


def main():
    data = clone_and_load_data()
    # plot_hist(data)
    pos_data = create_positive_examples(data)
    # don't use examples where last step is mistake
    pos_data = pos_data[pos_data["mistake_index"] < pos_data["num_steps"] - 1]

    # only save a subset of the columns
    pos_data = pos_data[
        ["input", "correct_steps", "incorrect_step", "mistake_index", "num_steps", "target", "task"]
    ]
    pos_data.rename(
        columns={
            "input": "question",
            "num_steps": "num_ground_truth_steps",
        },
        inplace=True,
    )

    # save data
    save_path = Path("evals/registry/data/error_recovery/main.jsonl")
    pos_data.to_json(save_path, orient="records", lines=True)

    small_save_path = Path("evals/registry/data/error_recovery/small.jsonl")
    # get small dataset with two examples from each task
    small_data = create_data_subset(pos_data, examples_per_task=2)
    small_data.to_json(small_save_path, orient="records", lines=True)

    medium_save_path = Path("evals/registry/data/error_recovery/medium.jsonl")
    # get medium dataset with 50 examples from each task
    medium_data = create_data_subset(pos_data, examples_per_task=50)
    medium_data.to_json(medium_save_path, orient="records", lines=True)


def create_data_subset(data: pd.DataFrame, examples_per_task: int) -> pd.DataFrame:
    # get small dataset with a subset of examples from each task
    small_data = pd.DataFrame()
    for task in data["task"].unique():
        task_data = data[data["task"] == task]
        task_subset = task_data[:examples_per_task]
        if len(task_subset) < examples_per_task:
            raise ValueError(
                f"Task {task} has only {len(task_subset)} examples, less than {examples_per_task}"
            )
        small_data = pd.concat((small_data, task_subset))
    return small_data


def create_positive_examples(data: pd.DataFrame) -> pd.DataFrame:
    has_incorrect_reasoning = ~data["mistake_index"].isnull()
    has_incorrect_answer = data["target"] != data["answer"]
    positive_condition = has_incorrect_reasoning & has_incorrect_answer

    positive_data = data.copy()
    positive_data = positive_data[positive_condition].reset_index()
    positive_data["label"] = "positive"
    positive_data["correct_steps"] = positive_data.apply(
        lambda row: row["steps"][: int(row["mistake_index"])], axis=1
    )
    positive_data["incorrect_step"] = positive_data.apply(
        lambda row: row["steps"][int(row["mistake_index"])], axis=1
    )
    return positive_data


def create_negative_examples(data: pd.DataFrame) -> pd.DataFrame:
    """Create a dataset of examples with correct reasoning and answer.

    The 'negative' naming is a bit misleading, but these are the examples
    we don't use.
    TODO (ian): think about renaming
    """
    has_correct_reasoning = data["mistake_index"].isnull()
    has_correct_answer = data["target"] == data["answer"]
    negative_condition = has_correct_reasoning & has_correct_answer
    negative_data = data.copy()
    negative_data = negative_data[negative_condition].reset_index()
    negative_data["label"] = "negative"
    negative_data["correct_steps"] = negative_data["steps"]
    negative_data["incorrect_step"] = ""
    return negative_data


def clone_and_load_data():
    clone_dir = Path("/tmp/BIG-Bench-Mistake")
    maybe_clone_repo(clone_dir)

    data = pd.DataFrame()
    for jsonl_file in clone_dir.glob("*.jsonl"):
        file_data = pd.read_json(jsonl_file, lines=True)

        # Manually append task description to datasets missing one
        task = jsonl_file.stem
        prefix = TASK_PREFIX[task]
        file_data["input"] = prefix + file_data["input"]
        file_data["task"] = task

        data = pd.concat((data, file_data))

    data["num_steps"] = data["steps"].apply(lambda x: len(x))
    return data


def maybe_clone_repo(clone_dir):
    if not clone_dir.exists():
        subprocess.run(
            ["git", "clone", "https://github.com/WHGTyen/BIG-Bench-Mistake.git", str(clone_dir)]
        )


def plot_hist(data):
    data["num_steps"].hist(bins=max(data["num_steps"]))
    plt.show()


def print_example():
    data = clone_and_load_data()
    # printing some examples
    subset_data = create_positive_examples(data)
    # subset_data = create_negative_examples(data)
    # # print one negative object swapping example
    # neg_example = neg_data[neg_data["task"] == "tracking_shuffled_objects"].iloc[0]
    # # print one negative dyck example
    # neg_example = neg_data[neg_data["task"] == "dyck_languages"].iloc[0]
    # neg_example = neg_data[neg_data["task"] == "logical_deduction"].iloc[0]
    example = subset_data[subset_data["task"] == "multistep_arithmetic"].iloc[1]
    print(f"INPUT ======\n{example['input']}")
    steps = "\n".join(example["steps"])
    print(f"STEPS ======\n{steps}")
    print(f"MISTAKE INDEX ======\n{example['mistake_index']}")
    print(f"ANSWER ======\n{example['answer']}")
    print(f"TARGET ======\n{example['target']}")
    print("========")


if __name__ == "__main__":
    main()
