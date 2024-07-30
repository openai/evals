import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm.auto import tqdm

from evals.utils import log_utils


def zero_if_none(input_num):
    if input_num is None:
        return 0
    else:
        return input_num


MODELS = [
    "gpt-4-0125-preview",
    "gpt-4-base",
    "gpt-4o",
    "gpt-3.5-turbo-0125",
    "gemini-pro-1.0",
    "mixtral-8x7b-instruct",
    "llama-2-70b-chat",
    "random_baseline",
    "human_baseline",
]
# separate list for OAI models for token counting, not supported in others.
OAI_MODELS = [
    "gpt-4-0125-preview",
    "gpt-3.5-turbo-0125",
    "gpt-4-base",
    "gpt-4o",
]

STAT_TO_LABEL = {
    "avg_max_length": "Average maximum sequence length achieved [no. of turns]",
    "violation_rate": "Violation rate",
}


def make_results_dict(log_dir: Path) -> dict:
    results_dict = prepare_results_dict()
    results_dict = fill_results_dict(results_dict, log_dir)
    return results_dict


def get_model(spec):
    # this is hilariously ugly but it works for now (sorry)
    if "gpt-4-turbo-preview" in spec["completion_fns"][0]:
        return "gpt-4-0125-preview"
    elif "gpt-3.5-turbo" in spec["completion_fns"][0]:
        return "gpt-3.5-turbo-0125"
    elif "gpt-4-base" in spec["completion_fns"][0]:
        return "gpt-4-base"
    elif "gpt-4o" in spec["completion_fns"][0]:
        return "gpt-4o"
    elif "gemini-pro" in spec["completion_fns"][0]:
        return "gemini-pro-1.0"
    elif "mixtral-8x7b-instruct" in spec["completion_fns"][0]:
        return "mixtral-8x7b-instruct"
    elif "llama-2-70b-chat" in spec["completion_fns"][0]:
        return "llama-2-70b-chat"
    elif "random_baseline" in spec["completion_fns"][0]:
        return "random_baseline"
    elif "human" in spec["completion_fns"][0]:
        return "human_baseline"


def get_state_tracking(spec):
    if "explicit" in spec["completion_fns"][0]:
        return "explicit"
    else:
        return "implicit"


def fill_results_dict(results_dict, log_dir):
    print("Parsing logs...")
    final_results = log_utils.get_final_results_from_dir(log_dir)
    specs = log_utils.get_specs_from_dir(log_dir)
    files = list(final_results.keys())

    for file in tqdm(files):
        final_result = final_results[file]
        spec = specs[file]
        task = spec["split"]
        model = get_model(spec)
        state_tracking = get_state_tracking(spec)
        for stat in results_dict:
            results_dict[stat][task][model][state_tracking]["raw"].append(final_result[stat])
    # compute means/std_errs
    for file in tqdm(files):
        spec = specs[file]
        task = spec["split"]
        model = get_model(spec)
        state_tracking = get_state_tracking(spec)
        for stat in results_dict:
            data_points = results_dict[stat][task][model][state_tracking]["raw"]
            results_dict[stat][task][model][state_tracking]["mean"] = np.mean(data_points)
            results_dict[stat][task][model][state_tracking]["std_err"] = np.std(
                data_points
            ) / np.sqrt(len(data_points) if len(data_points) > 1 else 1)
    return results_dict


def prepare_results_dict():
    results_dict = {
        stat: {
            task: {
                model: {state_tracking: {"raw": []} for state_tracking in ["implicit", "explicit"]}
                for model in MODELS
            }
            for task in ["mode", "median"]
        }
        for stat in ["avg_max_length", "violation_rate"]
    }
    return results_dict


def make_bar_plot(results_dict: dict, task: str, stat: str, save_path: Path):
    sns.set_context("paper")
    sns.set_style("whitegrid")

    data = results_dict[stat][task]

    # the random baseline and human baseline aren't plotted as bars
    models = MODELS[:-2]

    state_tracking_kinds = ["explicit", "implicit"]

    means = [[data[model][cat]["mean"] for cat in state_tracking_kinds] for model in models]
    std_errs = [[data[model][cat]["std_err"] for cat in state_tracking_kinds] for model in models]
    cmap = plt.get_cmap("Paired")
    colors = np.array([cmap(i) for i in range(len(state_tracking_kinds))])

    # Plotting
    x = np.arange(len(models))  # the label locations

    width = 0.4

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=300)

    explicit_bars = ax.barh(
        x + width / 2,
        [mean[0] for mean in means],
        width,
        xerr=[err[0] for err in std_errs],
        label="Explicitly tracked state baseline",
        color=colors[0],
    )
    implicit_bars = ax.barh(
        x - width / 2,
        [mean[1] for mean in means],
        width,
        xerr=[err[1] for err in std_errs],
        label="Implicitly tracked state",
        color=colors[1],
    )

    ax.set_xlabel(STAT_TO_LABEL[stat])
    # maximum x + xerr value times 1.2
    x_max = (
        max([m for mean in means for m in mean]) + max([e for err in std_errs for e in err])
    ) * 1.2
    ax.set_xlim([0, x_max])
    ax.set_yticks(x)
    ax.set_yticklabels(models)

    ax.bar_label(implicit_bars, padding=3, fmt="%.2f")
    ax.bar_label(explicit_bars, padding=3, fmt="%.2f")

    # plot random and human baselines
    random_baseline = data["random_baseline"]["implicit"]["mean"]
    random_err = data["random_baseline"]["implicit"]["std_err"]
    ax.axvline(random_baseline, color="red", linestyle="--", label="Random baseline")
    ax.axvspan(
        random_baseline - random_err,
        random_baseline + random_err,
        color="red",
        alpha=0.05,
    )

    human_baseline = data["human_baseline"]["implicit"]["mean"]
    human_err = data["human_baseline"]["implicit"]["std_err"]
    ax.axvline(
        human_baseline,
        color="#366a9d",
        linestyle=":",
        label="Human baseline (implicit)",
    )

    ax.axvspan(
        human_baseline - human_err,
        human_baseline + human_err,
        color="#366a9d",
        alpha=0.05,
    )

    # get rid of horizontal grid lines
    ax.grid(axis="y", which="both")

    ax.legend()

    fig.tight_layout()

    plt.savefig(save_path, bbox_inches="tight", dpi=300)


def count_tokens(log_dir) -> dict[str, dict[str, dict[str, int]]]:
    """
    model -> task -> input, output, total tokens
    """
    token_counts = {
        model: {
            task: {
                state_tracking: {kind: 0 for kind in ["input", "output", "total"]}
                for state_tracking in ["implicit", "explicit"]
            }
            for task in ["mode", "median"]
        }
        for model in OAI_MODELS
    }
    globbed_logs = list(log_dir.glob("*.log"))
    already_examined = set()
    for log in tqdm(globbed_logs, total=len(globbed_logs), desc="Counting tokens"):
        spec = log_utils.extract_spec(log)
        task = spec["split"]
        model = get_model(spec)
        state_tracking = get_state_tracking(spec)

        if model not in OAI_MODELS:
            continue

        # dont care about repeats, this is a rough estimate anyway
        if (model, task, state_tracking) in already_examined:
            continue
        already_examined.add((model, task, state_tracking))

        samplings = log_utils.extract_individual_results(log, "sampling")
        for sampling in samplings:
            usage = sampling["usage"]
            token_counts[model][task][state_tracking]["input"] += zero_if_none(
                usage["prompt_tokens"]
            )
            token_counts[model][task][state_tracking]["output"] += zero_if_none(
                usage["completion_tokens"]
            )
            token_counts[model][task][state_tracking]["total"] += zero_if_none(
                usage["total_tokens"]
            )
    return token_counts


def main(args: argparse.Namespace):
    log_dir = Path(args.log_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    results_dict = make_results_dict(log_dir)

    for stat in tqdm(results_dict.keys(), desc="Plotting..."):
        for task in tqdm(["mode", "median"], desc=f"Plotting {stat}"):
            save_path = save_dir / f"{task}_{stat}.png"
            make_bar_plot(results_dict, task, stat, save_path)
        save_path = save_dir / f"{stat}.json"
        with open(save_path, "w") as f:
            json.dump(results_dict[stat], f, indent=2)

    token_counts = count_tokens(log_dir)
    save_path = save_dir / "token_counts.json"
    with open(save_path, "w") as f:
        json.dump(token_counts, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True, help="Where the logs are stored")
    parser.add_argument("--save_dir", type=str, required=True, help="Where to save the plots")
    args = parser.parse_args()
    main(args)
