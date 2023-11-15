import argparse
import glob
import json
import os
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt


def load_jsonl_log(path):
    """
    Given jsonl log, returns list containing tuples
    of (question, was_answer_to_questopm_correct)
    """
    all_examples = []
    with open(path, "r") as f:
        for line in f:
            line = json.loads(line)
            if "spec" in line.keys() or "final_report" in line.keys() or line["type"] != "metrics":
                continue

            q = line["data"]["prompt"][2]["content"]

            correct = line["data"]["correct"]
            correct = int(correct)  # Cast to 0/1

            all_examples.append((q, correct))

    return all_examples


def get_questions(files: Sequence[Path]) -> dict:
    """
    Gets all questions. All runs have the same questions, so we
    pick one randomly
    """
    questions = {}

    # All runs have same Qs, so pick one randomly and extract Qs
    path = files[0]
    with open(path, "r") as f:
        for line in f:
            line = json.loads(line)
            if "spec" in line.keys() or "final_report" in line.keys() or line["type"] != "metrics":
                continue

            q = line["data"]["prompt"][2]["content"]
            questions[q] = []

    return questions


def get_answers_for_each_question(files: Sequence[Path], questions: dict):
    """
    Iterate over all runs, gathering all answers for each unique question
    """
    for path in files:
        with open(path, "r") as f:
            for line in f:
                line = json.loads(line)
                if (
                    "spec" in line.keys()
                    or "final_report" in line.keys()
                    or line["type"] != "metrics"
                ):
                    continue

                q = line["data"]["prompt"][2]["content"]

                correct = line["data"]["correct"]
                correct = int(correct)  # Cast to 0/1

                questions[q].append(correct)

    return questions


def create_total_no_correct_hist(files: Sequence[Path], questions: dict, outdir: Path):
    """
    Plot histogram of no. times each question was correctly answered
    """
    questions_total_correct = {k: sum(v) for k, v in questions.items()}
    all_totals = [i for i in questions_total_correct.values()]

    plt.hist(all_totals, bins=[i - 0.5 for i in range(len(files) + 2)], width=0.9)
    plt.xticks(range(len(files) + 1))
    plt.xlabel("No. correct answers for a question")
    plt.ylabel("Frequency")

    outpath = os.path.join(outdir, "consistency.png")
    plt.savefig(outpath)
    plt.show()


def calculate_consistency(files: Sequence[Path]):
    """
    Calculate the consistency over all runs. We take all pairs of runs and
    for each pair, calculate the proportion of questions that had the same
    answer. We average this over all pairs
    """
    all_consistencies = []
    for idx, i in enumerate(files):
        i_log = load_jsonl_log(i)
        for j in files[idx + 1 :]:
            j_log = load_jsonl_log(j)
            assert len(i_log) == len(j_log)

            # Get all answers for each question. Inefficient
            joined = []
            for q1, c1 in i_log:
                for q2, c2 in j_log:
                    if q1 == q2:
                        joined.append((q1, c1, c2))

            assert len(joined) == len(i_log), f"Len joined: {len(joined)}, Len j_log: {len(j_log)}"
            consistency = sum([c1 == c2 for _, c1, c2 in joined]) / len(joined)
            all_consistencies.append(consistency)

    consistency = sum(all_consistencies) / len(all_consistencies)
    print(f"Consistency: {consistency}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", "-d", type=str, required=True)
    parser.add_argument("--out_dir", "-o", type=str, default="./outputs")
    args = parser.parse_args()
    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    files = glob.glob(os.path.join(log_dir, "*.log"))
    questions = get_questions(files)
    questions = get_answers_for_each_question(files, questions)

    create_total_no_correct_hist(files, questions, out_dir)
    calculate_consistency(files)


if __name__ == "__main__":
    main()
