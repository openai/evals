import argparse
import os
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from evals.utils.log_utils import extract_spec, get_final_results_from_dir


def extract_results(datadir: Path) -> pd.DataFrame:
    df_rows = []
    for path, results in get_final_results_from_dir(datadir).items():
        spec = extract_spec(path)
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


def plot_results(df: pd.DataFrame, out_dir: Path, plot_horizontal: bool):
    models = df["model"].to_list()

    # Find all types of tools and bugs
    all_tools = []
    all_bugs = []
    for i in df.columns:
        if i.startswith("tool_") and i.endswith("f1"):
            all_tools.append(i)
        if i.startswith("bug_") and i.endswith("accuracy"):
            all_bugs.append(i)

    # Make ordering consistent
    all_tools.sort()
    all_bugs.sort()

    # Sort so tools are in ascending order of gpt-4 performance
    generic_gpt_4_solver = "generation/direct/gpt-4"
    if len([i for i in models if generic_gpt_4_solver == i]) == 1:
        gpt_4_row_idx = df.index[df["model"] == generic_gpt_4_solver][0]

        filtered_df = df[all_tools]
        filtered_df = filtered_df.sort_values(gpt_4_row_idx, axis=1)

        all_tools = []
        for i in filtered_df.columns:
            if i.startswith("tool_") and i.endswith("f1"):
                all_tools.append(i)

    # Plot results split by tool type
    results = {}
    for model in models:
        metrics = []
        for tool in all_tools:
            value = df[tool][df.model == model].item()
            value = str(value)
            if "%" in value:
                value = value.replace("%", "")
            value = float(value)
            metrics.append(value)

        results[model] = metrics

    all_tools_renamed = [i.split("tool_")[1].split("_f1")[0] for i in all_tools]

    plot_df = pd.DataFrame(results, index=all_tools_renamed)
    if plot_horizontal:
        plot_df.plot.barh(rot=0)
        plt.xlim(0, 1)
        plt.ylabel("Types of tools")
        plt.xlabel("F1")
    else:
        plot_df.plot.bar(rot=90)
        plt.ylim(0, 1)
        plt.xlabel("Types of tools")
        plt.ylabel("F1")

    outpath = os.path.join(out_dir, "results_split_by_tool.png")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.show()

    # Plot results split by bug type
    results = {}
    for model in models:
        metrics = []
        for bug in all_bugs:
            value = df[bug][df.model == model].item()
            value = str(value)
            if "%" in value:
                value = value.replace("%", "")
            value = float(value) * 100  # Accuracy in range [0, 100]
            metrics.append(value)

        results[model] = metrics

    all_bugs_renamed = [i.split("bug_")[1].split("_accuracy")[0] for i in all_bugs]
    plot_df = pd.DataFrame(results, index=all_bugs_renamed)
    if plot_horizontal:
        plot_df.plot.barh(rot=0)
        plt.xlim(0, 100)
        plt.ylabel("Types of bugs")
        plt.xlabel("Accuracy (%)")
    else:
        plot_df.plot.bar(rot=0)
        plt.ylim(0, 100)
        plt.xlabel("Types of bugs")
        plt.ylabel("Accuracy (%)")

    outpath = os.path.join(out_dir, "results_split_by_bug.png")
    plt.savefig(outpath)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", "-d", type=str, required=True)
    parser.add_argument("--out_dir", "-o", type=str, required=True)
    args = parser.parse_args()
    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    plot_horizontal = False

    df = extract_results(log_dir)
    plot_results(df, out_dir, plot_horizontal)


if __name__ == "__main__":
    main()
