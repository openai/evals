import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from evals.utils import log_utils

PLOT_TITLES_BY_METRIC = {
    "overall_accuracy": "Accuracy",  # ie. both retrieval and non-retrieval in one plot
    "baseline_accuracy": "Baseline accuracy (non-retrieval)",
    "retrieval_accuracy": "Retrieval accuracy",
    "average_retrieval_precision": "Average retrieval precision",
    "average_non_retrieval_bleu_score": "Average non-retrieval BLEU score",
    "average_retrieval_bleu_score": "Average retrieval BLEU score",
    "average_retrieval_calls": "Average retrieval calls",
    "average_invalid_retrieval_calls": "Average invalid retrieval calls",
    "bleu_score": "BLEU score",
    "correct_call_rate": "Correct call rate",
    "invalid_call_rate": "Invalid call rate",
    "timeout_rate": "Timeout rate",
    "ctx_len_exceeded_rate": "Context length exceeded rate",
}

UNIT_METRICS = set(
    ["correct_call_rate", "invalid_call_rate", "timeout_rate", "ctx_len_exceeded_rate"]
)


def extract_metrics(datadir: Path) -> pd.DataFrame:
    df_rows = []
    for path, results in sorted(list(log_utils.get_final_results_from_dir(datadir).items())):
        spec = log_utils.extract_spec(path)
        solver_path = Path(spec["completion_fns"][0])
        model = solver_path.name
        solver = solver_path.parent.name
        # Remove root section of path, which is the eval name
        solver_path = solver_path.relative_to(solver_path.parts[0])
        df_rows.append({"solver": solver, "model": model, **results})
    df = pd.DataFrame(df_rows)

    return df


def make_plot(
    df: pd.DataFrame,
    outpath: Path,
    metric="baseline_accuracy",
    min_ylim=0,
    max_ylim=0.08,
    dataset="miskito",
):
    plt.figure()
    sns.set_theme(style="whitegrid")
    # Calculating mean and SEM
    grouped = df.groupby(["model", "solver"])[metric].agg(["mean", "sem"]).reset_index()

    def compute_sem(x):
        sem = x.std() / (len(x) ** 0.5)
        sem2 = sem * 2  # 95% confidence interval
        return (x.mean() - sem2, x.mean() + sem2)

    # Plotting
    sns.set(style="whitegrid")
    sns.barplot(x="model", y="mean", hue="solver", data=grouped, errorbar=compute_sem, capsize=0.1)
    plt.xticks(rotation=30, ha="right")
    plt.ylim(min_ylim, max_ylim)

    # Some of the metrics are in [0, 1].
    if metric in UNIT_METRICS:
        plt.ylim(0, 1)

    plt.title(PLOT_TITLES_BY_METRIC[metric] + f" on {dataset.capitalize()} Q&A dataset")
    plt.xlabel("Model")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def make_side_bar_plot(
    df: pd.DataFrame,
    outpath: Path,
    metric="overall_accuracy",
    min_ylim=0,
    max_ylim=0.1,
    dataset="miskito",
):
    if metric == "overall_accuracy":
        df_clean = df[["model", "solver", "baseline_accuracy", "retrieval_accuracy"]]
    elif metric == "bleu_score":
        df_clean = df[
            ["model", "solver", "average_non_retrieval_bleu_score", "average_retrieval_bleu_score"]
        ]

    fig, ax = plt.subplots(figsize=(10, 5))
    # df_clean = df_clean.drop(columns=["solver"])
    df_clean.set_index(["model", "solver"], inplace=True)

    # Group by 'model' and calculate mean and SEM
    grouped = df_clean.groupby(["model", "solver"]).agg(["mean", "sem"])
    xlabels = [f"{model}/{solver}" for model, solver in grouped.index]

    # Prepare data for plotting
    means = grouped.xs("mean", axis=1, level=1)
    errors = grouped.xs("sem", axis=1, level=1)

    # Plotting
    means.plot(kind="bar", yerr=errors, capsize=4, ax=ax)  # Removed 'stacked=True'

    ax.set_ylabel(metric)
    ax.set_xticklabels(xlabels, rotation=30, ha="right")
    ax.set_xlabel("model/solver")
    ax.set_ylim(min_ylim, max_ylim)

    fig.tight_layout(pad=3.0)
    fig.suptitle(PLOT_TITLES_BY_METRIC[metric] + f" on {dataset.capitalize()} dataset")
    fig.savefig(outpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", "-d", type=str, required=True)
    parser.add_argument("--out-dir", "-o", type=str, default="./outputs")
    args = parser.parse_args()
    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(exist_ok=True, parents=True)

    datasets = os.listdir(log_dir)

    for dataset in datasets:
        print(f"Extracting data for eval dataset {dataset}...")
        df = extract_metrics(log_dir / dataset)

        # Rename some of the solver values so they can be represented in the same plot.
        df.loc[df["solver"] == "cot_hhh", "solver"] = "cot"
        df.loc[df["solver"] == "hhh", "solver"] = "direct"
        df.loc[df["solver"] == "fewshot_direct", "solver"] = "fewshot"

        # TODO: report directly as 'average_correct_calls' in future and remove this rename.
        df.rename(columns={"average_retrieval_precision": "average_correct_calls"}, inplace=True)
        df["correct_call_rate"] = df["average_correct_calls"] / df["average_retrieval_calls"]
        df["invalid_call_rate"] = (
            df["average_invalid_retrieval_calls"] / df["average_retrieval_calls"]
        )

        print(f"Plotting other metrics for eval dataset {dataset}...")

        # Generate bar plots for all other metrics.
        core_metrics = (
            []
        )  # ["baseline_accuracy", "retrieval_accuracy", "average_non_retrieval_bleu_score", "average_retrieval_bleu_score"]
        auxiliary_metrics = [
            "correct_call_rate",
            "invalid_call_rate",
            "timeout_rate",
            "ctx_len_exceeded_rate",
        ]
        for metric in core_metrics + auxiliary_metrics:
            make_plot(
                df[["model", "solver", metric]].copy(),
                out_dir / f"{dataset}_{metric}.png",
                metric,
                dataset=dataset,
            )

        print(f"Plotting headline metrics for eval dataset {dataset}...")

        # Generate stacked bar plots for the two headline metrics.
        for metric in ["overall_accuracy", "bleu_score"]:
            make_side_bar_plot(df, out_dir / f"{dataset}_{metric}.png", metric, dataset=dataset)

        # Print numerical results (and compute % improvement metrics)
        grouped = df.groupby(["model", "solver"]).agg(["mean", "sem"])
        for type, closedbook, openbook in [
            (
                "Translation (BLEU)",
                "average_non_retrieval_bleu_score",
                "average_retrieval_bleu_score",
            ),
            ("Non-translation (%)", "baseline_accuracy", "retrieval_accuracy"),
        ]:
            print(f"Improvement Metrics for {type} on {dataset.capitalize()} dataset")
            improvement_rows = []
            for idx, row in grouped.iterrows():
                openbook_score = row[openbook]["mean"]
                closedbook_score = row[closedbook]["mean"]
                rel_improvement_score = (openbook_score - closedbook_score) / (1 - closedbook_score)
                improvement_rows.append(
                    {
                        "model": idx[0],
                        "solver": idx[1],
                        "closedbook": closedbook_score,
                        "openbook": openbook_score,
                        "improvement": rel_improvement_score,
                    }
                )
            improvement_df = pd.DataFrame(improvement_rows)
            print(improvement_df)
            # print to stdout as csv
            print(improvement_df.to_csv(index=False))
