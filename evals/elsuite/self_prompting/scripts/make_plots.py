import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dataset.eval_list import eval_list

from evals.utils import log_utils


def extract_metrics(datadir: Path) -> pd.DataFrame:
    df_rows = []
    for path, results in sorted(list(log_utils.get_final_results_from_dir(datadir).items())):
        spec = log_utils.extract_spec(path)
        solver_path = Path(spec["completion_fns"][0])
        model = solver_path.name
        solver = solver_path.parent.name
        # Remove root section of path, which is the eval name
        solver_path = solver_path.relative_to(solver_path.parts[0])
        for res in log_utils.extract_individual_results(path):
            df_rows.append(
                {
                    "solver_path": solver_path,
                    "model": model,
                    "solver": solver,
                    "taskname": res["task"]["eval"],
                    **res,
                }
            )
    df = pd.DataFrame(df_rows)
    # Sort rows
    df = df.sort_values(by=["model", "solver", "taskname", "tasker_model"])

    # Add rows with tasker_model="mean"
    df_all = df.copy()
    df_all["tasker_model"] = "mean"

    df = pd.concat([df, df_all])
    return df


def make_plot(df: pd.DataFrame, outpath: Path, metric="exact"):
    sns.set_theme(style="whitegrid")

    df = df[df["tasker_model"] == "mean"]

    def compute_sem(x):
        sem = x.std() / (len(x) ** 0.5)
        sem2 = sem * 2  # 95% confidence interval
        return (x.mean() - sem2, x.mean() + sem2)

    # Plot mean+sem accuracy, grouped by model and solver
    sns.pointplot(
        data=df,
        x="model",
        y=metric,
        hue="solver",
        errorbar=compute_sem,  # Use standard error of the mean
        dodge=True,  # Separate points for different hues
        capsize=0.1,  # Caps for the error bars
        errwidth=1,  # Width of the error bars
        markers=".",  # Marker style
        linestyles="",  # No line connecting the points
    )
    plt.legend(loc="upper right", ncol=2)
    # Rotate x-axis labels, align end to center
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1)

    plt.title(f"Mean tasker accuracy ({metric})")
    plt.xlabel("Prompter")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", "-d", type=str, required=True)
    parser.add_argument("--out_dir", "-o", type=str, default="./outputs")
    args = parser.parse_args()
    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(exist_ok=True, parents=True)

    metrics_df = extract_metrics(log_dir)

    # Our results are an average over different task distributions, handle with care
    if set(metrics_df["taskname"].unique()) != set(eval_list):
        print(
            "WARNING: Task distribution changed, results and error bars will not be comparable to plots with the original task distribution."
        )

    # Sample a subset of the data for inspection
    subset_df = metrics_df[metrics_df["tasker_model"] != "mean"]
    # Take only the first row of each [solver_path, taskname, tasker_model] group
    subset_df = subset_df.groupby(["solver_path", "taskname", "tasker_model"]).first().reset_index()
    subset_df.to_csv(out_dir / "subset_samples.csv", quoting=csv.QUOTE_ALL, escapechar="\\")

    make_plot(metrics_df, out_dir / "per_tasker_results_exact.png", metric="exact")
    make_plot(metrics_df, out_dir / "per_tasker_results_fuzzy.png", metric="fuzzy")

    # Print results
    exact_df_rows = []
    fuzzy_df_rows = []
    violation_df_rows = []
    for _, df_tasker in metrics_df.groupby(["model", "solver"]):
        solver = df_tasker["solver"].iloc[0]
        model = df_tasker["model"].iloc[0]

        exact = df_tasker.groupby("tasker_model")["exact"].mean()
        exact_df_rows.append(
            {
                "model": model,
                "solver": solver,
                **exact,
            }
        )
        fuzzy = df_tasker.groupby("tasker_model")["fuzzy"].mean()
        fuzzy_df_rows.append(
            {
                "model": model,
                "solver": solver,
                **fuzzy,
            }
        )
        prompt_rule_violation = df_tasker.groupby("tasker_model")["prompt_rule_violation"].mean()
        violation_df_rows.append(
            {
                "model": model,
                "solver": solver,
                **prompt_rule_violation,
            }
        )

    exact_df = pd.DataFrame(exact_df_rows)
    exact_df.to_csv(out_dir / "exact.csv", quoting=csv.QUOTE_ALL, index=False)
    print(exact_df)
    fuzzy_df = pd.DataFrame(fuzzy_df_rows)
    fuzzy_df.to_csv(out_dir / "fuzzy.csv", quoting=csv.QUOTE_ALL, index=False)
    print(fuzzy_df)
    violation_df = pd.DataFrame(violation_df_rows)
    violation_df.to_csv(out_dir / "violation.csv", quoting=csv.QUOTE_ALL, index=False)
    print(violation_df)


if __name__ == "__main__":
    main()
