from typing import Dict, Iterable, List
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


renderers_of_interest = ["csv", "language-corrset"]

renderer_to_label = {
    "csv": "CSV observations",
    "language-corrset": "Correlation set",
}

cmap = plt.get_cmap("Paired")
colors = np.array([cmap(i) for i in range(len(renderers_of_interest))])
renderer_to_color = {r: c for r, c in zip(renderers_of_interest, colors)}

solver_to_label = {
    "generation/direct/gpt-3.5-turbo": "Direct gpt-3.5-turbo",
    "generation/cot/gpt-3.5-turbo": "CoT gpt-3.5-turbo",
    "generation/hhh/gpt-4-base": "HHH gpt-4-base",
    "generation/cot_hhh/gpt-4-base": "CoT HHH gpt-4-base",
    "generation/direct/gpt-4-1106-preview": "Direct gpt-4-1106-preview",
    "generation/cot/gpt-4-1106-preview": "CoT gpt-4-1106-preview",
    "generation/cot/mixtral-8x7b-instruct": "CoT mixtral-8x7b-instruct\n(Correlation set only)",
    "generation/cot/llama-2-70b-chat": "CoT llama-2-70b-chat\n(Correlation set only)",
    "generation/cot/gemini-pro": "CoT gemini-pro-1.0\n(Correlation set only)",
    "identifying_variables/random": "Random baseline",
    "identifying_variables/noctrl": "NoCtrl baseline",
}

baseline_to_linestyle = {
    "identifying_variables/random": "--",
    "identifying_variables/noctrl": "-.",
}

cmap = plt.get_cmap("Set2")
bline_colors = np.array(
    [cmap(i) for i in range(0, len(baseline_to_linestyle.keys()) + 0)]
)
baseline_to_color = {
    key: color for key, color in zip(baseline_to_linestyle.keys(), bline_colors)
}


def plot_solver_bars(
    bar_solvers: List[str],
    baseline_solvers: List[str],
    metric_results: Dict,
    metric_label: str,
    fig_height: int,
    output_path: Path,
):
    """
    Plots a side-by-side bar plot of the metric results, showing the
    solvers on the x axis and the metric value on the y axis.

    Args:
        bar_solvers: The names of solvers to plot.
        baseline_solvers: The names of the baseline solvers to plot.
        metric_results: A dictionary with k: v of format solver : {mean: value, sem: value}
        metric_label: The label for the y axis
        fig_height: the height of the figure in inches
        output_path: the path to save the figure to
    """
    sns.set_context("paper")
    sns.set_style("whitegrid")

    bar_width = 0.3
    positions = np.arange(len(bar_solvers))

    f, ax = plt.subplots(1, 1, dpi=300, figsize=(9, fig_height))

    for i, renderer in enumerate(renderers_of_interest):
        bars = [
            metric_results["mean"][solver][renderer]["without tree"]
            for solver in bar_solvers
        ]
        errors = [
            metric_results["sem"][solver][renderer]["without tree"]
            for solver in bar_solvers
        ]

        ax.bar(
            positions + bar_width * i,
            bars,
            bar_width,
            yerr=errors,
            label=renderer_to_label[renderer],
            color=renderer_to_color[renderer],
        )

    for baseline_solver in baseline_solvers:
        mean = metric_results["mean"][baseline_solver]["corrset"]["without tree"]
        sem = metric_results["sem"][baseline_solver]["corrset"]["without tree"]
        ax.axhline(
            mean,
            label=solver_to_label[baseline_solver],
            color=baseline_to_color[baseline_solver],
            linestyle=baseline_to_linestyle[baseline_solver],
        )
        ax.axhspan(
            mean - sem, mean + sem, alpha=0.1, color=baseline_to_color[baseline_solver]
        )

    ax.set_xticks(
        positions + bar_width / 2,
        [solver_to_label[s] for s in bar_solvers],
        rotation=45,
        ha="right",
    )
    ax.tick_params(
        axis="x", which="both", bottom=True
    )  # Show both major and minor xticks
    ax.set_ylabel(metric_label)
    ax.set_ylim(-0.005, 1)
    ax.xaxis.grid(False)
    ax.legend()
    f.set_tight_layout(True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")


def plot_difficulty_bars(results_dict: Dict, bins: Iterable[int], output_path: Path):
    sns.set_context("paper")
    sns.set_style("whitegrid")

    f, ax = plt.subplots(1, 1, dpi=300, figsize=(7, 4))

    positions = np.arange(len(bins))
    bar_width = 0.4

    for i, key in enumerate(sorted(results_dict.keys())):
        solver, renderer = key.split(";")
        bars = [results_dict[key][bbin]["mean"] for bbin in bins]
        errors = [results_dict[key][bbin]["sem"] for bbin in bins]
        if solver == "generation/direct/gpt-4-1106-preview":
            label = renderer_to_label[renderer]
            color = renderer_to_color[renderer]
            ax.bar(
                positions + bar_width * i,
                bars,
                bar_width,
                yerr=errors,
                label=label,
                color=color,
            )

    ax.set_xlabel("Number of necessary control variables")
    ax.set_ylabel("Control Variable Retrieval nDCG*")

    ax.set_xlim(-0.3, 8.7)
    ax.set_ylim(0, 1)
    ax.xaxis.grid(False)
    ax.legend()
    ax.set_xticks(positions + bar_width / 2, bins)
    f.set_tight_layout(True)
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
