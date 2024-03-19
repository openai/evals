import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from evals.utils import log_utils

# MODEL_NAMES = {
#     "error_recovery/gpt-4-0613": "GPT-4",
#     "generation/hhh/gpt-4-base": "GPT-4 Base",
#     "error_recovery/gpt-3.5-turbo-0613": "GPT-3.5",
#     # "gpt-4-base": "gpt-4-base",
# }
# using model checkpoint names
MODEL_NAMES = {
    "error_recovery/gpt-4-0613": "gpt-4-0613",
    "generation/hhh/gpt-4-base": "gpt-4-base",
    "error_recovery/gpt-3.5-turbo-0613": "gpt-3.5-turbo-0613",
    # "generation/direct/llama-2-13b-chat": "llama-2-13b-chat",
    "generation/direct/llama-2-70b-chat": "llama-2-70b-chat",
    "generation/direct/mixtral-8x7b-instruct": "mixtral-8x7b-instruct",
    "generation/direct/gemini-pro": "gemini-pro-1.0",
}

MODEL_COLOR_MAP = {
    "error_recovery/gpt-4-0613": "purple",
    "generation/hhh/gpt-4-base": "plum",
    "error_recovery/gpt-3.5-turbo-0613": "g",
    # "generation/direct/llama-2-13b-chat": "wheat",
    "generation/direct/llama-2-70b-chat": "orange",
    "generation/direct/mixtral-8x7b-instruct": "red",
    "generation/direct/gemini-pro": "cornflowerblue",
}
VARIATION_NAMES = {
    "nr_name": "From Scratch",
    "cr_name": "Correct Basis",
    "ir_name": "Incorrect Basis",
}

VARIATION_COLOR_MAP = {
    "nr_name": "blue",
    "cr_name": "green",
    "ir_name": "red",
}

TASK_NAMES = {
    "word_sorting": "Word Sorting",
    "tracking_shuffled_objects": "Tracking Shuffled Objects",
    "logical_deduction": "Logical Deduction",
    "multistep_arithmetic": "Multi-Step Arithmetic",
    "dyck_languages": "Dyck Languages",
}


def maybe_show(fig):
    if DISPLAY:
        fig.show()
    plt.close(fig)


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


def extract_metrics(datadir: Path) -> pd.DataFrame:
    df_rows = []
    for path, results in sorted(list(log_utils.get_final_results_from_dir(datadir).items())):
        spec = log_utils.extract_spec(path)
        solver = spec["completion_fns"][0]
        for res in log_utils.extract_individual_results(path):
            df_rows.append(
                {
                    "solver": solver,
                    **res,
                }
            )
    df = pd.DataFrame(df_rows)
    # Sort rows
    # print(df.columns)
    df.sort_values(by=["solver", "task"], inplace=True)
    return df


def get_all_tasks(results_df: pd.DataFrame) -> list[str]:
    # Find all types of tasks
    all_tasks = []
    for i in results_df.columns:
        if i.startswith("task_") and i.endswith("_CR_correct_rate"):
            all_tasks.append(i)

    # Make ordering consistent
    all_tasks.sort()
    return all_tasks


def get_all_tasks_renamed(results_df: pd.DataFrame) -> list[str]:
    all_tasks = get_all_tasks(results_df)
    all_tasks_renamed = [i.split("task_")[1].split("_CR_correct_rate")[0] for i in all_tasks]
    # replace hyphens with underscores
    all_tasks_renamed = [i.replace("-", "_") for i in all_tasks_renamed]
    return all_tasks_renamed


def get_unique_models(results_df: pd.DataFrame) -> list[str]:
    models = results_df["model"].to_list()
    # TODO: work out how to order a variable set of models
    if set(models) == set(MODEL_NAMES.keys()):
        unique_models = list(MODEL_NAMES.keys())
    else:
        unique_models = sorted(list(set(models)), reverse=True)
    return unique_models


def get_cleaned_model_name(model: str) -> str:
    return model.replace("/", "_")


def corrects_to_accuracy_and_sem(corrects: pd.Series):
    accuracy = corrects.mean()
    sem = corrects.sem()
    return accuracy, sem


def annotate_axes(ax, errors: Optional[pd.DataFrame]):
    """Annotate each bar in the plot with its value"""
    ABOVE_OFFSET = 0.01
    BELOW_OFFSET = 0.1
    if errors is not None:
        # This gets it into a shape to match the order of the patch objects.
        # I don't have a principled reason to transpose, this is just what works.
        error_values = errors.to_numpy().T.flatten()

    for i, p in enumerate(ax.patches):
        # patch objects aren't typed correctly
        p_height = p.get_height()  # type: ignore
        p_x = p.get_x()  # type: ignore
        p_width = p.get_width()  # type: ignore
        # Calculate the label position
        x = p_x + p_width / 2
        if errors is not None:
            error = error_values[i]
        else:
            error = 0

        if p_height > 0:
            y = p_height + error + ABOVE_OFFSET
        else:
            y = p_height - error - BELOW_OFFSET

        # Annotate the bar with its value
        # ax.annotate(f"{p_height:.2f}\n±{error:.2f}", (x, y), ha="center", va="bottom")
        ax.annotate(f"{p_height:.2f}", (x, y), ha="center", va="bottom")


def corrects_to_performance_loss_and_error(CR_corrects: pd.Series, IR_corrects: pd.Series):
    CR_correct_rate = CR_corrects.mean()
    IR_correct_rate = IR_corrects.mean()

    performance_recovered = IR_correct_rate / CR_correct_rate
    performance_loss = 1 - performance_recovered
    # propagate error from CR_corrects and IR_corrects to performance_loss
    CR_correct_rate_sem = CR_corrects.sem()
    IR_correct_rate_sem = IR_corrects.sem()
    assert isinstance(CR_correct_rate_sem, float)
    assert isinstance(IR_correct_rate_sem, float)
    # using the formula for error propagation for a ratio from
    # https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Example_formulae
    # (assuming errors in CR and IR are independent).
    # NOTE: the 1 in performance_loss is a constant,
    # so doesn't affect the uncertainty bounds on the ratio.
    CR_term = (CR_correct_rate_sem / CR_correct_rate) ** 2
    IR_term = (IR_correct_rate_sem / IR_correct_rate) ** 2
    performance_loss_error = abs(performance_recovered) * ((CR_term + IR_term) ** 0.5)
    print(f"Performance loss: {performance_loss:.2f} ± {performance_loss_error:.2f}")
    return performance_loss, performance_loss_error


def accuracy_by_task(metrics_df, results_df: pd.DataFrame, out_dir: Path):
    all_tasks = get_all_tasks(results_df)
    unique_models = get_unique_models(results_df)
    all_tasks_renamed = get_all_tasks_renamed(results_df)

    # Plot results separately for each model
    for model in unique_models:
        plot_accuracy_by_task(model, metrics_df, all_tasks, all_tasks_renamed, out_dir)


def accuracy_by_model_dfs(metrics_df, results_df: pd.DataFrame):
    unique_models = get_unique_models(results_df)
    accuracies = {}
    sems = {}
    for model in unique_models:
        pass
        # for all tasks
        model_mask = metrics_df.solver == model
        model_CR_corrects = metrics_df[model_mask]["CR_correct"]
        model_IR_corrects = metrics_df[model_mask]["IR_correct"]
        model_NR_corrects = metrics_df[model_mask]["NR_correct"]

        model_CR_accuracy, model_CR_sem = corrects_to_accuracy_and_sem(model_CR_corrects)
        model_IR_accuracy, model_IR_sem = corrects_to_accuracy_and_sem(model_IR_corrects)
        model_NR_accuracy, model_NR_sem = corrects_to_accuracy_and_sem(model_NR_corrects)

        pretty_model_name = MODEL_NAMES[model]
        sems[pretty_model_name] = {
            "nr_name": model_NR_sem,
            "cr_name": model_CR_sem,
            "ir_name": model_IR_sem,
        }
        accuracies[pretty_model_name] = {
            "nr_name": model_NR_accuracy,
            "cr_name": model_CR_accuracy,
            "ir_name": model_IR_accuracy,
        }

    order = ["nr_name", "cr_name", "ir_name"]
    plot_df = pd.DataFrame(accuracies)
    plot_df = plot_df.reindex(order)
    sems_df = pd.DataFrame(sems)
    sems_df = sems_df.reindex(order)
    return plot_df, sems_df


def accuracy_by_model(metrics_df, results_df: pd.DataFrame, out_dir: Path):
    unique_models = get_unique_models(results_df)
    plot_df, sems_df = accuracy_by_model_dfs(metrics_df, results_df)

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    colors = [MODEL_COLOR_MAP[model] for model in unique_models]
    plot_df.index = list(VARIATION_NAMES.values())
    sems_df.index = list(VARIATION_NAMES.values())
    ax = plot_df.plot.bar(
        rot=0,
        yerr=sems_df,
        capsize=4,
        ax=ax,
        width=0.8,
        color=colors,
    )
    annotate_axes(ax, sems_df)
    ax.set_ylim(top=1.0)
    ax.set_xlabel("Reasoning variations")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy for each variation (higher is better)")

    outpath = os.path.join(out_dir, "accuracy_by_model.png")
    fig.savefig(outpath)
    maybe_show(fig)


def accuracy_by_model_and_reasoning(
    own_metrics_df: pd.DataFrame,
    own_results_df: pd.DataFrame,
    other_metrics_df: pd.DataFrame,
    other_results_df: pd.DataFrame,
    out_dir: Path,
):
    own_plot_df, own_sems_df = accuracy_by_model_dfs(own_metrics_df, own_results_df)
    other_plot_df, other_sems_df = accuracy_by_model_dfs(other_metrics_df, other_results_df)
    # drop the no reasoning baseline
    own_plot_df = own_plot_df.drop("nr_name", axis=0)
    own_sems_df = own_sems_df.drop("nr_name", axis=0)
    other_plot_df = other_plot_df.drop("nr_name", axis=0)
    other_sems_df = other_sems_df.drop("nr_name", axis=0)

    own_plot_df = own_plot_df.T
    own_sems_df = own_sems_df.T
    other_plot_df = other_plot_df.T
    other_sems_df = other_sems_df.T
    models = own_plot_df.index  # e.g., ["No reasoning (baseline)", "Correct reasoning", ...]
    n_models = len(models)
    bar_width = 0.35  # width of the bars
    n_variations = len(own_plot_df.columns)
    assert n_variations == len(other_plot_df.columns)
    group_width = 0.8  # Total width for one group of bars
    bar_width = group_width / (n_variations * 2)  # Width of one bar

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)

    # Set position of bar on X axis
    ind = np.arange(n_models)  # the x locations for the groups

    colors = [VARIATION_COLOR_MAP[variation] for variation in own_plot_df.columns]
    VARIATION_OFFSET = 0.03
    for i, variation in enumerate(own_plot_df.columns):
        # Position of bars for this model
        # bars for a given model are grouped together, and then within that group, the bars for each variation are grouped
        r1 = ind + i * VARIATION_OFFSET + i * (n_variations * bar_width)
        r2 = [x + bar_width for x in r1]

        ax.bar(
            r1,
            own_plot_df[variation],
            width=bar_width,
            yerr=own_sems_df[variation],
            capsize=5,
            label=f"{VARIATION_NAMES[variation]} ('assistant' message)",
            color=colors[i],
            # add outline to bars
            edgecolor="black",
        )
        ax.bar(
            r2,
            other_plot_df[variation],
            width=bar_width,
            yerr=other_sems_df[variation],
            capsize=5,
            label=f"{VARIATION_NAMES[variation]} ('user' message)",
            hatch="//",
            color=colors[i],
            edgecolor="black",
        )

        for j, model in enumerate(models):
            x_own = r1[j]
            x_other = r2[j]
            y1 = own_plot_df.loc[model, variation]
            y2 = other_plot_df.loc[model, variation]
            y1_err = own_sems_df.loc[model, variation]
            y2_err = other_sems_df.loc[model, variation]
            ax.text(x_own, y1 + y1_err, f"{y1:.2f}", ha="center", va="bottom")
            ax.text(x_other, y2 + y2_err, f"{y2:.2f}", ha="center", va="bottom")

    # Add xticks on the middle of the group bars
    xtick_positions = ind + bar_width * n_variations + (VARIATION_OFFSET - bar_width) / 2
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(models)

    # Create legend & Show graphic
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(top=1.0)
    ax.legend()
    ax.set_title("Accuracy for each variation (higher is better)")
    outpath = os.path.join(out_dir, "accuracy_by_category_and_reasoning.png")
    fig.savefig(outpath)
    maybe_show(fig)


def plot_accuracy_by_steps_all(metrics_df, results_df, out_dir):
    """
    Create plots of accuracy of:
        - num_steps - mistake_index
        - mistake_index / num_steps
    """
    get_all_tasks(results_df)
    all_tasks_renamed = get_all_tasks_renamed(results_df)
    all_models = get_unique_models(results_df)
    # one plot per task, one subplot per model
    for task in all_tasks_renamed:
        fig, axs = plt.subplots(
            1, len(all_models), figsize=(15, 6), constrained_layout=True, squeeze=False
        )
        axs = axs.flatten()
        for ax, model in zip(axs, all_models):
            task_model_df = metrics_df[(metrics_df.solver == model) & (metrics_df.task == task)]
            plot_accuracy_by_steps(task_model_df, task, model, ax)
        # only put legend on last plot
        final_ax = axs[-1]
        final_ax.legend(loc="upper center")
        outpath = os.path.join(out_dir, f"results-split-by-steps_{task}.png")
        fig.suptitle(f"Accuracy by steps for {TASK_NAMES[task]} (higher is better)")
        fig.savefig(outpath)
        maybe_show(fig)


def plot_accuracy_by_steps(df, task, model, ax):
    df["steps_diff"] = df["num_ground_truth_steps"] - df["mistake_index"]

    # due to the way pandas works, we have to group, then filter, then regroup
    grouped_df = df.groupby("steps_diff")

    MIN_SAMPLES = 10
    filtered_groups = grouped_df.filter(lambda x: len(x) >= MIN_SAMPLES)

    # Now, re-group the filtered DataFrame by 'steps_diff' again and calculate the mean
    plot_df = filtered_groups.groupby("steps_diff")[
        ["NR_correct", "CR_correct", "IR_correct"]
    ].mean()
    colors = [VARIATION_COLOR_MAP[variation] for variation in VARIATION_NAMES.keys()]

    # change the names of the columns to be more readable before plotting
    plot_df.columns = list(VARIATION_NAMES.values())
    # now plot the three accuracies against steps_diff
    assert isinstance(plot_df, pd.DataFrame)
    ax = plot_df.plot(color=colors, ax=ax, legend=False)
    ax.set_xlabel("Steps beyond mistake")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.1)
    # ax.set_title(f"{MODEL_NAMES[model]} | {TASK_NAMES[task]} (higher is better)")
    ax.set_title(f"{MODEL_NAMES[model]}")
    # plt.tight_layout()
    return ax


def plot_accuracy_by_task(model, metrics_df, all_tasks, all_tasks_renamed, out_dir):
    all_tasks_pretty = [TASK_NAMES[i] for i in all_tasks_renamed]
    accuracies = {"nr_name": [], "cr_name": [], "ir_name": []}
    all_sems = []
    # for all tasks
    model_mask = metrics_df.solver == model

    # and split by task type
    for task in all_tasks_renamed:

        task_mask = metrics_df.task == task
        CR_corrects = metrics_df[model_mask & task_mask]["CR_correct"]
        IR_corrects = metrics_df[model_mask & task_mask]["IR_correct"]
        NR_corrects = metrics_df[model_mask & task_mask]["NR_correct"]

        CR_accuracy, CR_sem = corrects_to_accuracy_and_sem(CR_corrects)
        IR_accuracy, IR_sem = corrects_to_accuracy_and_sem(IR_corrects)
        NR_accuracy, NR_sem = corrects_to_accuracy_and_sem(NR_corrects)

        accuracies["nr_name"].append(NR_accuracy)
        accuracies["cr_name"].append(CR_accuracy)
        accuracies["ir_name"].append(IR_accuracy)

        sems = [NR_sem, CR_sem, IR_sem]
        all_sems.append(sems)

    sems_df = pd.DataFrame(
        all_sems,
        index=all_tasks_pretty,
        columns=["nr_name", "cr_name", "ir_name"],
    )

    plot_df = pd.DataFrame(accuracies, index=all_tasks_pretty)

    fig, ax = plt.subplots(figsize=(15, 6), constrained_layout=True)
    colors = [VARIATION_COLOR_MAP[variation] for variation in plot_df.columns]
    plot_df.columns = list(VARIATION_NAMES.values())
    ax = plot_df.plot.bar(rot=0, color=colors, yerr=sems_df, capsize=4, ax=ax, width=0.8)
    annotate_axes(ax, sems_df)

    # Shrink current axis by 20% to make room for the legend
    box = ax.get_position()
    ax.set_position((box.x0, box.y0, box.width * 0.8, box.height))
    # Place the legend outside the plot
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_ylim(top=1.1)
    ax.set_xlabel("Task type")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{MODEL_NAMES[model]} (higher is better)")
    outpath = os.path.join(out_dir, f"results-split-by-task_{get_cleaned_model_name(model)}.png")
    fig.savefig(outpath)
    maybe_show(fig)


def performance_loss_per_task(metrics_df: pd.DataFrame, results_df: pd.DataFrame, out_dir: Path):
    # Plot performance lost for each model
    unique_models = get_unique_models(results_df)
    get_all_tasks(results_df)
    all_tasks_renamed = get_all_tasks_renamed(results_df)
    all_tasks_pretty = [TASK_NAMES[i] for i in all_tasks_renamed]

    all_metrics = {}
    all_errors = {}
    for model in unique_models:
        metrics = []
        errors = []
        for task in all_tasks_renamed:
            model_mask = metrics_df.solver == model
            task_mask = metrics_df.task == task
            CR_corrects = metrics_df[model_mask & task_mask]["CR_correct"]
            IR_corrects = metrics_df[model_mask & task_mask]["IR_correct"]

            performance_loss, performance_loss_error = corrects_to_performance_loss_and_error(
                CR_corrects, IR_corrects
            )
            metrics.append(performance_loss)
            errors.append(performance_loss_error)

        pretty_model_name = MODEL_NAMES[model]
        all_metrics[pretty_model_name] = metrics
        all_errors[pretty_model_name] = errors

    fig, ax = plt.subplots(figsize=(20, 6), constrained_layout=True)
    plot_df = pd.DataFrame(all_metrics, index=all_tasks_pretty)
    errs_df = pd.DataFrame(all_errors, index=all_tasks_pretty)
    colors = [MODEL_COLOR_MAP[model] for model in unique_models]
    ax = plot_df.plot.bar(rot=0.0, color=colors, ax=ax, width=0.8, yerr=errs_df, capsize=4)
    annotate_axes(ax, errs_df)
    # Shrink current axis by 20% to make room for the legend
    box = ax.get_position()
    ax.set_position((box.x0, box.y0, box.width * 0.8, box.height))
    ax.set_ylim(bottom=-1, top=1.1)
    ax.legend()
    ax.axhline(0, 0, 1, color="black", linestyle="-")
    ax.set_title("Performance loss per task (lower is better)")
    ax.set_xlabel("Task type")
    ax.set_ylabel("Performance loss")

    outpath = os.path.join(out_dir, "results_split_by_model.png")
    fig.savefig(outpath)
    maybe_show(fig)


def performance_loss_per_model(metrics_df: pd.DataFrame, results_df: pd.DataFrame, out_dir: Path):
    unique_models = get_unique_models(results_df)

    metrics = {}
    errors = {}
    for model in unique_models:
        model_mask = metrics_df.solver == model

        CR_corrects = metrics_df[model_mask]["CR_correct"]
        IR_corrects = metrics_df[model_mask]["IR_correct"]

        performance_loss, performance_loss_error = corrects_to_performance_loss_and_error(
            CR_corrects, IR_corrects
        )

        pretty_model_name = MODEL_NAMES[model]
        metrics[pretty_model_name] = performance_loss
        errors[pretty_model_name] = performance_loss_error

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    plot_df = pd.DataFrame(metrics, index=[0])
    errs_df = pd.DataFrame(errors, index=[0])
    colors = [MODEL_COLOR_MAP[model] for model in unique_models]
    ax = plot_df.plot.bar(rot=0, color=colors, ax=ax, width=0.8, yerr=errs_df, capsize=4)
    annotate_axes(ax, errs_df)
    # Shrink current axis by 20% to make room for the legend
    box = ax.get_position()
    ax.set_position((box.x0, box.y0, box.width * 0.8, box.height))
    # Place the legend outside the plot
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_ylabel("Performance loss")
    ax.set_ylim(top=1.1)
    ax.set_title("Average performance loss per model (lower is better)")
    outpath = os.path.join(out_dir, "headline_results.png")
    fig.savefig(outpath)
    maybe_show(fig)


def main():
    parser = argparse.ArgumentParser()
    # DEBUG: hacking together own_reasoning and other_reasoning plots
    parser.add_argument(
        "--log_dir",
        "-d",
        type=str,
        required=True,
        help="Path to log dir with primary results (if supplementary_dir is provided, this is should be 'own' reasoning)",
    )
    parser.add_argument(
        "--supplementary_dir",
        "-s",
        type=str,
        help="Optional supplementary log dir with 'other' reasoning results",
    )
    parser.add_argument("--out_dir", "-o", type=str, required=True)
    args = parser.parse_args()
    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    metrics_df = extract_metrics(log_dir)
    results_df = extract_results(log_dir)
    if args.supplementary_dir:
        other_log_dir = Path(args.supplementary_dir)
        other_metrics_df = extract_metrics(other_log_dir)
        other_results_df = extract_results(other_log_dir)
        accuracy_by_model_and_reasoning(
            metrics_df, results_df, other_metrics_df, other_results_df, out_dir
        )
    accuracy_by_task(metrics_df, results_df, out_dir)
    accuracy_by_model(metrics_df, results_df, out_dir)
    performance_loss_per_task(metrics_df, results_df, out_dir)
    performance_loss_per_model(metrics_df, results_df, out_dir)
    plot_accuracy_by_steps_all(metrics_df, results_df, out_dir)


if __name__ == "__main__":
    DISPLAY = False
    main()
