import os
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

from evals.elsuite.sandbagging.utils import compute_sandbagging_score
from evals.utils import log_utils


def extract_results(datadir: Path) -> pd.DataFrame:
    df_rows = []
    for path, results in log_utils.get_final_results_from_dir(datadir).items():
        spec = log_utils.extract_spec(path)
        model = spec["completion_fns"][0]
        base_eval = spec["base_eval"]
        df_rows.append(
            {
                "model": model,
                "base_eval": base_eval,
                **results,
            }
        )
    df = pd.DataFrame(df_rows)
    return df


def compute_all_model_scores(model_log: dict, thresh_targets: bool) -> float:
    """
    Given log, calculate sandbagging score for each model
    """
    # Find min baseline
    if thresh_targets:
        baselines = [model_log[name]["baseline"] for name in model_log.keys()]
        min_baseline = min(baselines)
    else:
        min_baseline = None

    # Compute MAE, up to min baseline
    model_scores = {}
    for name, v in model_log.items():
        targets = v["targets"]
        accs = v["accs"]
        score = compute_sandbagging_score(targets, accs, min_baseline)
        model_scores[name] = score

    return model_scores


def make_sandbagging_plot(
    df: pd.DataFrame,
    out_dir: str,
    file_name: str,
    title: str,
    colors: Sequence[str],
    baseline_eval_name: str,
    sb_eval_name: str,
    plot_all_data_together=False,
):
    """
    Generic plot for sandbagging, parses given DataFrame to extract indicated
    data.
    """

    def finish_plot(outpath):
        plt.plot([0, 100], [0, 100], label="Perfect sandbagging", c="purple")

        plt.xlim([-2, 102])
        plt.ylim([0, 102])
        plt.xlabel("Target accuracy")
        plt.ylabel("Actual accuracy")
        plt.title(title)
        plt.legend()

        plt.savefig(outpath)
        plt.show()

    # Get all unique model names. Format solver names to get model name out
    model_names = []
    models = df["model"].unique()
    for name in models:
        formatted_name = name.split("/")[-1]
        model_names.append(formatted_name)

    model_names = list(set(model_names))

    # Get accuracy and std column names
    cols = df.columns
    acc_cols = [i for i in cols if "accuracy_target_" in i]
    std_cols = [i for i in cols if "bootstrap_std_target_" in i]

    # Get all unique targets that were sandbagged to
    targets = [i.split("target_")[1].split("%")[0] for i in acc_cols]
    targets = [float(i) for i in targets]

    # Plot accuracies and errors for every model, for every target
    model_log = {}
    for name, c in zip(model_names, colors):
        df_model = df[df.model.str.contains(name)]

        # Plot zero-shot baseline
        baseline = df_model[df_model.base_eval == baseline_eval_name]
        baseline = baseline["accuracy"].values[0] * 100
        plt.plot(
            [0, 100],
            [baseline, baseline],
            label=f"{name} no sandbagging baseline",
            linestyle="dashed",
            c=c,
        )

        # Gather accuracies and stds
        df_model_sb = df_model[df_model.base_eval == sb_eval_name]

        all_accuracies = [df_model_sb[col].values[0] * 100 for col in acc_cols]
        all_stds = [df_model_sb[col].values[0] * 100 for col in std_cols]

        # Plot accuracies and stds
        plt.errorbar(
            targets,
            all_accuracies,
            yerr=all_stds,
            fmt="o",
            c=c,
            ecolor="black",
            capsize=5,
            label=f"{name} sandbagging",
        )

        # Log model sandbagged accuracies
        model_log[name] = {}
        model_log[name]["baseline"] = baseline
        model_log[name]["targets"] = targets
        model_log[name]["accs"] = all_accuracies

        if not plot_all_data_together:
            finish_plot(outpath=os.path.join(out_dir, f"{name}_{file_name}"))

    if plot_all_data_together:
        finish_plot(outpath=os.path.join(out_dir, file_name))

    return model_log
