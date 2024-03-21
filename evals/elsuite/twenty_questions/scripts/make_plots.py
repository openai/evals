import argparse
from pathlib import Path
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from evals.utils import log_utils

PLOT_TITLES_BY_METRIC = {
    "score": "Score",
    "winrate": "Win-rate",
    "bootstrap_std": "Bootstrapped standard deviation",
    "average_num_guesses": "Average guesses per sample",
    "average_num_questions": "Average questions per sample",
    "average_num_violations": "Average rule violations per sample",
    "average_num_gamemaster_refusals": "Average gamemaster refusals per sample",
    "average_num_incorrect_guesses": "Average incorrect guesses per sample",
    "average_word_difficulty": "Average word difficulty",
}

HUMAN_BASELINE = {
    "standard": {
        "winrate": 0.0333,
        "score": 0.1333,
        "average_num_guesses": 0.3666,
        "average_num_questions": 19.8666,
        "average_num_violations": 0.62,
        "average_num_gamemaster_refusals": 0.28,
        "average_num_incorrect_guesses": 0.3333,
        "average_word_difficulty": 2.2333,
    },
    "shortlist": {
        "winrate": 1,
        "score": 14.1388,
        "average_num_guesses": 1.8611,
        "average_num_questions": 5.8611,
        "average_num_violations": 0.1944,
        "average_num_gamemaster_refusals": 0.1111,
        "average_num_incorrect_guesses": 0.8611,
        "average_word_difficulty": 2.2777,
    }
}

UNIT_METRICS = ["winrate"]

def extract_metrics(datadir: Path) -> pd.DataFrame:
    df_rows = []
    # There are two eval variants: standard and shortlist.
    for variant in os.listdir(datadir):
        for path, results in sorted(list(log_utils.get_final_results_from_dir(f"{datadir}/{variant}").items())):
            spec = log_utils.extract_spec(path)
            solver_path = Path(spec["completion_fns"][0])
            model = solver_path.name
            solver = solver_path.parent.name
            # Remove root section of path, which is the eval name
            solver_path = solver_path.relative_to(solver_path.parts[0])
            df_rows.append({"solver": solver, "model": model, "variant": variant, **results})
    df = pd.DataFrame(df_rows)
    df.rename(columns={"accuracy": "winrate"}, inplace=True)
    df.sort_values(by=["variant", "model", "solver"], inplace=True)
    df.to_csv(datadir / "results.csv", index=False)

    return df

def make_plot(df: pd.DataFrame, outpath: Path, metric="score", variant="standard"):
    df = df.round(2)
    plt.figure()
    sns.set_theme(style="whitegrid")

    def compute_sem(x):
        sem = x.std() / (len(x) ** 0.5)
        sem2 = sem * 2  # 95% confidence interval
        lower = max(0, (x.mean() - sem2).round(2))
        upper = (x.mean() + sem2).round(2)
        return lower, upper


    # Plotting
    sns.set(style="whitegrid")
    ax = sns.barplot(x=metric, y="model", hue="solver", data=df, errorbar=compute_sem, capsize=0.1)
    for container in ax.containers:
        ax.bar_label(container, fmt="{:.2f}", label_type="edge", padding=15)
    
    ax.axvline(HUMAN_BASELINE[variant][metric], color="red", linestyle="--")

    # A bunch of tweaks to make individual plots look nice.
    if variant == "shortlist" and metric == "winrate":
        plt.text(HUMAN_BASELINE[variant][metric] - 0.35, .5, "Human baseline", color="red", fontsize=12, ha="left")
    elif variant == "standard" and metric == "average_num_questions":
        plt.text(HUMAN_BASELINE[variant][metric] - 7, .5, "Human baseline", color="red", fontsize=12, ha="left")
    else:
        plt.text(HUMAN_BASELINE[variant][metric] + 0.05, .5, "Human baseline", color="red", fontsize=12, ha="left")

    # Some of the metrics are in [0, 1].
    if metric in UNIT_METRICS:
        plt.xlim(0, 1.1)

    if metric in ("score", "average_num_questions"):
        plt.xlim(0, 20.1)

    if metric == "average_word_difficulty":
        plt.xlim(0, 3.1)  # 6 is the maximum word difficulty in the dataset.

    if metric in ("score", "winrate"):
        plt.legend(loc="lower right")

    plt.title(PLOT_TITLES_BY_METRIC[metric] + f" ({variant} variant)")
    plt.xlabel(metric)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", "-d", type=str, required=True)
    parser.add_argument("--out-dir", "-o", type=str, default="./outputs")
    args = parser.parse_args()
    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(exist_ok=True, parents=True)

    df = extract_metrics(log_dir)

    # Rename some of the solver values so they can be represented in the same plot.
    df.loc[df['solver'] == 'cot_hhh', 'solver'] = 'cot'
    df.loc[df['solver'] == 'hhh', 'solver'] = 'direct'

    for variant in df['variant'].unique():
        df_per_variant = df[df['variant'] == variant]

        print(f"Plotting all metrics for {variant} variant...")

        core_metrics = ["score", "winrate"]
        auxiliary_metrics = ["average_num_guesses", "average_num_questions", "average_num_violations", "average_num_gamemaster_refusals", "average_num_incorrect_guesses", "average_word_difficulty"]
        for metric in core_metrics + auxiliary_metrics:
            make_plot(df_per_variant[["model", "solver", metric]].copy(), 
                    out_dir / f"{variant}_{metric}.png", 
                    metric,
                    variant)