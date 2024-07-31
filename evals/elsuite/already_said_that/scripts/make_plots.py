from pathlib import Path
import argparse
import json

from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from evals.utils import log_utils


def zero_if_none(input_num):
    if input_num is None:
        return 0
    else:
        return input_num


MODELS = [
    "cot/gpt-4-turbo-preview",
    "gpt-4-turbo-preview",
    "cot/gpt-3.5-turbo",
    "gpt-3.5-turbo",
    "gpt-4-base",
    "gpt-4o",
    "gemini-pro",
    "mixtral-8x7b-instruct",
    "llama-2-70b-chat",
    "random_baseline",
]
# separate list for OAI models for token counting, not supported in others.
OAI_MODELS = [
    "cot/gpt-4-turbo-preview",
    "gpt-4-turbo-preview",
    "cot/gpt-3.5-turbo",
    "gpt-3.5-turbo",
    "gpt-4-base",
    "gpt-4o",
]


DISTRACTORS = [
    "which-is-heavier",
    "ambiguous-sentences",
    "first-letters",
    "reverse-sort-words-eng",
    "distractorless",
]


MODEL_TO_LABEL = {
    "cot/gpt-4-turbo-preview": "CoT gpt-4-0125-preview",
    "cot/gpt-3.5-turbo": "CoT gpt-3.5-turbo-0125",
    "gpt-4-turbo-preview": "Direct gpt-4-0125-preview",
    "gpt-3.5-turbo": "Direct gpt-3.5-turbo-0125",
    "gpt-4-base": "HHH gpt-4-base",
    "gemini-pro": "Direct gemini-pro-1.0",
    "mixtral-8x7b-instruct": "Direct mixtral-8x7b-instruct",
    "llama-2-70b-chat": "Direct llama-2-70b-chat",
    "random_baseline": "Random Baseline",
}

NUM_REPEATS = 3

PLOT_STATS = ["avg_num_turns", "avg_distractor_accuracy"]
JSON_STATS = [
    "avg_num_turns",
    "avg_distractor_accuracy",
    "false_positive_rate",
    "false_negative_rate",
    "violation_rate",
]

STAT_TO_MAX = {
    "avg_num_distractors": 100 / 3,  # distractors shown every 1/3 of the time
    "avg_num_turns": 100,  # best case, we run out of steps
    "avg_distractor_accuracy": 1,
    "false_positive_rate": 1,
    "false_negative_rate": 1,
    "violation_rate": 1,
}

STAT_TO_LABEL = {
    "avg_num_distractors": "Average number of distractors shown before failure",
    "avg_num_turns": "Average number of turns before failure",
    "avg_distractor_accuracy": "Average accuracy on distractor task",
    "false_positive_rate": "False positive rate",
    "false_negative_rate": "False negative rate",
    "violation_rate": "Violation rate",
}


def make_results_dict(log_dir: Path) -> dict:
    results_dict = prepare_results_dict()
    results_dict = fill_results_dict(results_dict, log_dir)
    return results_dict


def prepare_results_dict() -> dict:
    results_dict = {
        stat: {
            distractor: {
                model: {"raw": [], "mean": 0, "std_err": 0} for model in MODELS
            }
            for distractor in DISTRACTORS
        }
        for stat in [
            "avg_num_distractors",
            "avg_num_turns",
            "avg_distractor_accuracy",
            "false_positive_rate",
            "false_negative_rate",
            "violation_rate",
        ]
    }
    return results_dict


def fill_results_dict(results_dict: dict, log_dir: Path) -> dict:
    print("Parsing logs...")
    final_results = log_utils.get_final_results_from_dir(log_dir)
    specs = log_utils.get_specs_from_dir(log_dir)
    files = list(final_results.keys())

    for file in tqdm(files):
        final_result = final_results[file]
        spec = specs[file]
        distractor = spec["split"]
        model = get_model(spec)
        for stat in results_dict:
            results_dict[stat][distractor][model]["raw"].append(final_result[stat])
    for file in tqdm(files):
        spec = specs[file]
        distractor = spec["split"]
        model = get_model(spec)
        # compute means/std_errs
        for stat in results_dict:
            data_points = results_dict[stat][distractor][model]["raw"]
            results_dict[stat][distractor][model]["mean"] = np.mean(data_points)
            results_dict[stat][distractor][model]["std_err"] = np.std(
                data_points
            ) / np.sqrt(NUM_REPEATS)
    return results_dict


def get_model(spec):
    # this is hilariously ugly but it works for now (sorry)
    if "cot/gpt-4-turbo-preview" in spec["completion_fns"][0]:
        return "cot/gpt-4-turbo-preview"
    elif "gpt-4-turbo-preview" in spec["completion_fns"][0]:
        return "gpt-4-turbo-preview"
    elif "cot/gpt-3.5-turbo" in spec["completion_fns"][0]:
        return "cot/gpt-3.5-turbo"
    elif "gpt-3.5-turbo" in spec["completion_fns"][0]:
        return "gpt-3.5-turbo"
    elif "gpt-4-base" in spec["completion_fns"][0]:
        return "gpt-4-base"
    elif "gpt-4o" in spec["completion_fns"][0]:
        return "gpt-4o"
    elif "gemini-pro" in spec["completion_fns"][0]:
        return "gemini-pro"
    elif "mixtral-8x7b-instruct" in spec["completion_fns"][0]:
        return "mixtral-8x7b-instruct"
    elif "llama-2-70b-chat" in spec["completion_fns"][0]:
        return "llama-2-70b-chat"
    elif "random_baseline" in spec["completion_fns"][0]:
        return "random_baseline"


def make_bar_plot(results_dict: dict, stat: str, save_path: Path):
    sns.set_context("paper")
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(1, 1, figsize=(8, 7), dpi=300)

    data = results_dict[stat]

    # the random baseline isn't plotted as bars
    models = MODELS[:-1]

    distractors = [
        "which-is-heavier",
        "ambiguous-sentences",
        "first-letters",
        "reverse-sort-words-eng",
    ]

    width = 0.15
    if stat != "avg_distractor_accuracy":
        distractors.append("distractorless")
        diffs = [-width * 2, -width / 1, 0, width / 1, width * 2]
        ax.axvline(STAT_TO_MAX[stat], label="maximum", linestyle="--", color="grey")

        # random baseline is roughly the same for all distractors; pick one for simplicity
        random_baseline = data["first-letters"]["random_baseline"]["mean"]

        ax.axvline(
            random_baseline,
            label=MODEL_TO_LABEL["random_baseline"],
            linestyle="-.",
            color="black",
        )

        # make legend order match bar order, idk why matplotlib reverses them
        legend_indices = [0, 1, 6, 5, 4, 3, 2]
    else:
        diffs = [-width * 1.5, -width / 2, width / 2, width * 1.5]
        legend_indices = list(range(len(distractors)))[::-1]

    means = [[data[dis][model]["mean"] for dis in distractors] for model in models]
    std_errs = [
        [data[dis][model]["std_err"] for dis in distractors] for model in models
    ]
    cmap = plt.get_cmap("Set3")
    colors = np.array([cmap(i) for i in range(len(distractors))])

    x = np.arange(len(models))  # the label locations

    distractor_bars = []
    for i, distractor in enumerate(distractors):
        bar = ax.barh(
            x + diffs[i],
            [mean[i] for mean in means],
            width,
            xerr=[err[i] for err in std_errs],
            label=distractor,
            color=colors[i] if distractor != "distractorless" else "black",
        )
        distractor_bars.append(bar)

    ax.set_xlabel(STAT_TO_LABEL[stat])
    x_max = STAT_TO_MAX[stat] + 0.05 * STAT_TO_MAX[stat]
    ax.set_xlim([0, x_max])
    ax.set_yticks(x)
    ax.set_yticklabels([MODEL_TO_LABEL[model] for model in models])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        [handles[i] for i in legend_indices],
        [labels[i] for i in legend_indices],
        loc="best",
    )

    for bar, distractor in zip(distractor_bars, distractors):
        ax.bar_label(
            bar,
            label_type="edge",
            fmt="%.2f",
            # color="white" if distractor == "distractorless" else "black",
            fontsize=8,
        )

    # get rid of horizontal grid lines
    ax.grid(axis="y", which="both")

    fig.set_tight_layout(True)

    plt.savefig(save_path, bbox_inches="tight", dpi=300)


def count_tokens(log_dir) -> dict[str, dict[str, dict[str, int]]]:
    """
    model -> distractor -> input, output, total tokens
    """
    token_counts = {
        model: {
            distractor: {kind: 0 for kind in ["input", "output", "total"]}
            for distractor in DISTRACTORS
        }
        for model in OAI_MODELS
    }
    globbed_logs = list(log_dir.glob("*.log"))
    already_examined = set()
    for log in tqdm(globbed_logs, total=len(globbed_logs), desc="Counting tokens"):
        spec = log_utils.extract_spec(log)
        distractor = spec["split"]
        model = get_model(spec)
        if model not in OAI_MODELS:
            continue

        # dont care about repeats, this is a rough estimate anyway
        if (model, distractor) in already_examined:
            continue
        already_examined.add((model, distractor))

        samplings = log_utils.extract_individual_results(log, "sampling")
        for sampling in samplings:
            usage = sampling["usage"]
            token_counts[model][distractor]["input"] += zero_if_none(
                usage["prompt_tokens"]
            )
            token_counts[model][distractor]["output"] += zero_if_none(
                usage["completion_tokens"]
            )
            token_counts[model][distractor]["total"] += zero_if_none(
                usage["total_tokens"]
            )
    return token_counts


def main(args: argparse.Namespace):
    log_dir = Path(args.log_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    results_dict = make_results_dict(log_dir)

    for stat in tqdm(PLOT_STATS, desc="Making plots"):
        save_path = save_dir / f"{stat}.png"
        make_bar_plot(results_dict, stat, save_path)

    for stat in tqdm(JSON_STATS, desc="Saving JSONs"):
        save_path = save_dir / f"{stat}.json"
        with open(save_path, "w") as f:
            json.dump(results_dict[stat], f, indent=2)

    token_counts = count_tokens(log_dir)
    save_path = save_dir / "token_counts.json"
    with open(save_path, "w") as f:
        json.dump(token_counts, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_dir", type=str, required=True, help="Where the logs are stored"
    )
    parser.add_argument(
        "--save_dir", type=str, required=True, help="Where to save the plots"
    )
    args = parser.parse_args()
    main(args)
