"""Take results from recent experiments and make a bar plot"""
import argparse
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from evals.utils import log_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    log_dir = args.log_dir
    out_dir = args.out_dir
    df = load_tom_results_from_dir(log_dir)
    make_plot(df, out_dir=Path(out_dir))


def load_tom_results_from_dir(log_dir: Union[str, Path]) -> pd.DataFrame:
    rows = []
    final_results_dict = log_utils.get_final_results_from_dir(log_dir)

    for path, final_results in final_results_dict.items():
        spec = log_utils.extract_spec(path)
        dataset, prompt_type, model = parse_spec(spec)
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "prompt_type": prompt_type,
                "accuracy": final_results["accuracy"],
                "bootstrap_std": final_results["bootstrap_std"],
            }
        )
    return pd.DataFrame(rows)


def parse_spec(spec: dict) -> tuple[str, bool, int]:
    """parse the spec from a MMP run"""
    completion_fn = spec["completion_fns"][0]
    dataset, prompt_type, model = completion_fn.split("/")
    prompt_type = prompt_type.split("_")[0]

    return (dataset, prompt_type, model)


def make_plot(df, out_dir):
    sns.set_theme(style="whitegrid")
    sns.set_palette("dark")
    # Define the order of models
    model_order = ["gpt-3.5-turbo", "gpt-4-base", "gpt-4"]
    datasets = df["dataset"].unique()

    for dataset in datasets:
        ds = df[df["dataset"] == dataset.lower()]

        # Ensure the model column is a categorical type with the specified order
        ds["model"] = pd.Categorical(ds["model"], categories=model_order, ordered=True)
        ds = ds.sort_values("model")  # Sort according to the categorical order

        # Unique models
        xs = ds["model"].unique()
        # Get the accuracy values for both prompt types
        simple_acc = ds[ds["prompt_type"] == "simple"]["accuracy"].values
        cot_acc = ds[ds["prompt_type"] == "cot"]["accuracy"].values

        # Get the corresponding error values from the "bootstrap_std" field
        simple_std = ds[ds["prompt_type"] == "simple"]["bootstrap_std"].values
        cot_std = ds[ds["prompt_type"] == "cot"]["bootstrap_std"].values

        # Define the width of a bar
        bar_width = 0.35
        # Set the positions of the bars
        x_indices = np.arange(len(xs))
        x_indices2 = [x + bar_width for x in x_indices]

        fig, ax1 = plt.subplots()
        fig.suptitle(f"Accuracy on {dataset} dataset")

        ax1.set_xlabel("Model")
        ax1.set_ylabel("Accuracy")

        # Plot the bars for 'simple' and 'cot'
        ax1.bar(
            x_indices,
            simple_acc,
            width=bar_width,
            color=sns.color_palette("pastel")[0],
            yerr=simple_std,
            label="simple",
        )
        ax1.bar(
            x_indices2,
            cot_acc,
            width=bar_width,
            color=sns.color_palette("pastel")[1],
            yerr=cot_std,
            label="chain-of-thought",
        )

        if dataset == "socialiqa":
            # Draw the horizontal line for the human baseline
            human_baseline = 0.881
            ax1.axhline(y=human_baseline, color="gray", linestyle="--", linewidth=1)
            # Add the text label for the human baseline
            ax1.text(
                0.01, human_baseline, "human baseline", va="center", ha="left", backgroundcolor="w"
            )

        # Set the x-axis ticks to be in the middle of the two bars
        ax1.set_xticks([r + bar_width / 2 for r in range(len(xs))])
        ax1.set_xticklabels(xs, rotation=45)  # Rotate the x-axis labels if needed

        ax1.set_ylim(0, 1)

        # Add legend
        ax1.legend(loc="upper right", bbox_to_anchor=(1, 1))

        # Save the figure
        plt.savefig(out_dir / f"accuracy_{dataset.lower()}.png", bbox_inches="tight")
        plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
        plt.show()


if __name__ == "__main__":
    main()
