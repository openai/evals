"""Take results from recent experiments and make a bar plot"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evals.utils import log_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    log_dir = args.log_dir
    out_dir = args.out_dir
    df = load_bp_results_from_dir(log_dir)
    make_plots(df, out_dir=Path(out_dir))


def load_bp_results_from_dir(log_dir: str) -> pd.DataFrame:
    rows = []
    final_results_dict = log_utils.get_final_results_from_dir(log_dir)
    for path in final_results_dict.keys():
        individual_results = log_utils.extract_individual_results(path)
        bp_metrics = compute_bp_metrics_from_individual_results(individual_results)

        spec = log_utils.extract_spec(path)
        influencer_model, voter_model, n_interactions = parse_spec(spec)

        rows.append(
            {
                "influencer_model": influencer_model,
                "voter_model": voter_model,
                "n_interactions": n_interactions,
                "influencer_targets": bp_metrics["influencer_target_list"],
                "voted_yes": bp_metrics["voted_yes_list"],
                "is_success": bp_metrics["is_success_list"],
            }
        )
    return pd.DataFrame(rows)


def compute_bp_metrics_from_individual_results(results):
    influencer_target_list = []
    voted_yes_list = []
    is_success_list = []
    for result in results:
        if "is_vote_yes__target_yes" in result:
            influencer_target_list.append("Yes")
            voted_yes_list.append(result["is_vote_yes__target_yes"])
            is_success_list.append(result["is_success"])
        elif "is_vote_yes__target_no" in result:
            influencer_target_list.append("No")
            voted_yes_list.append(result["is_vote_yes__target_no"])
            is_success_list.append(result["is_success"])
        elif "is_vote_yes__target_control" in result:
            influencer_target_list.append("Control")
            voted_yes_list.append(result["is_vote_yes__target_control"])
            is_success_list.append(None)
        else:
            raise ValueError(f"Unexpected result {result}")
    return {
        "influencer_target_list": influencer_target_list,
        "voted_yes_list": voted_yes_list,
        "is_success_list": is_success_list,
    }


def parse_spec(spec: dict) -> tuple[str, str, int]:
    """parse the spec from a bp run"""
    voter_model = spec["completion_fns"][0]
    influencer_model = spec["completion_fns"][1]
    n_interactions = int(spec["run_config"]["eval_spec"]["args"]["n_interactions"])

    assert influencer_model in [
        "gpt-3.5-turbo-16k",
        "gpt-4",
        "gpt-4-base",
    ], f"Unexpected influencer_model: {influencer_model}"
    assert voter_model in [
        "gpt-4",
    ], f"Unexpected voter_model: {voter_model} (experiments assume gpt-4)"
    assert n_interactions in [3, 5], f"Unexpected n_interactions: {n_interactions}"
    return influencer_model, voter_model, n_interactions


def make_plots(df: pd.DataFrame, out_dir: Path):
    """Take results from experiments and make plots"""
    out_dir.mkdir(exist_ok=True, parents=True)
    make_yes_ratio_by_target_plot(df)
    plt.savefig(out_dir / "yes_ratio_by_target.png")
    plt.show()
    make_yes_ratio_by_model_plot(df)
    plt.savefig(out_dir / "yes_ratio_by_model.png")
    plt.show()
    make_success_rate_plot(df)
    plt.savefig(out_dir / "success_rate.png")
    plt.show()


def make_yes_ratio_by_model_plot(df):
    bars_dict = extract_vote_data_from_df(df)
    _make_model_plot(bars_dict)


def make_yes_ratio_by_target_plot(df):
    """assumes that the only voter model is gpt-4, as in the report"""
    bars_dict = extract_vote_data_from_df(df)
    _make_target_plot(bars_dict)


def make_success_rate_plot(df):
    bars_dict = extract_vote_data_from_df(df)
    _make_success_plot(bars_dict)


def extract_vote_data_from_df(df):
    bars_dict = {}
    for _, row in df.iterrows():
        target_dict = {}
        influencer_model = row["influencer_model"]
        n_interactions = row["n_interactions"]
        influencer_targets = row["influencer_targets"]
        voted_yes = row["voted_yes"]
        is_success = row["is_success"]
        for target, vote, success in zip(influencer_targets, voted_yes, is_success):
            if target not in target_dict:
                target_dict[target] = {"votes": [], "successes": []}
            target_dict[target]["votes"].append(vote)
            target_dict[target]["successes"].append(success)
        if (influencer_model, n_interactions) in bars_dict:
            raise ValueError(f"Duplicate key {(influencer_model, n_interactions)}")
        bars_dict[(influencer_model, n_interactions)] = {
            "Yes": {
                "votes_mean": np.mean(target_dict["Yes"]["votes"]),
                "votes_sem": np.std(target_dict["Yes"]["votes"], ddof=1)
                / np.sqrt(len(target_dict["Yes"]["votes"])),
                "successes": target_dict["Yes"]["successes"],
            },
            "Control": {
                "votes_mean": np.mean(target_dict["Control"]["votes"]),
                "votes_sem": np.std(target_dict["Control"]["votes"], ddof=1)
                / np.sqrt(len(target_dict["Control"]["votes"])),
            },
            "No": {
                "votes_mean": np.mean(target_dict["Yes"]["votes"]),
                "votes_sem": np.std(target_dict["Yes"]["votes"], ddof=1)
                / np.sqrt(len(target_dict["Yes"]["votes"])),
                "successes": target_dict["No"]["successes"],
            },
        }
    return bars_dict


def _make_model_plot(bars_dict):
    # Sort keys for a consistent ordering in the legend
    sorted_keys = sorted(bars_dict.keys(), key=lambda x: (x[0], x[1]))

    vote_types = ["Yes", "Control", "No"]
    increment = 1.5
    positions = np.arange(0, len(vote_types) * increment, increment)
    bar_width = 0.15

    fig, ax = plt.subplots(figsize=(12, 6))

    # Iterate through each (model, number) pair and plot bars for each vote type
    for i, key in enumerate(sorted_keys):
        means = [bars_dict[key][vote]["votes_mean"] for vote in vote_types]
        sems = [bars_dict[key][vote]["votes_sem"] for vote in vote_types]

        ax.bar(
            positions + i * bar_width,
            means,
            bar_width,
            label=f"{key[0]}, {key[1]}",
            yerr=sems,
            capsize=5,
        )

    ax.set_xticks(positions + bar_width * (len(sorted_keys) - 1) / 2)
    ax.set_xticklabels(vote_types)
    ax.set_ylabel("Mean Vote Percentage")
    ax.set_xlabel("Vote Type")
    ax.set_title("Mean Vote Percentage by Vote Type with (Model, Number) in Legend")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

    ax.yaxis.grid(True, linestyle="--", which="major", color="grey", alpha=0.25)
    plt.tight_layout()


def _make_target_plot(bars_dict):
    # Sort keys for a consistent ordering on the x-axis
    sorted_keys = sorted(bars_dict.keys(), key=lambda x: (x[0], x[1]))

    labels = [f"{key[0]}, {key[1]}" for key in sorted_keys]

    yes_means = []
    no_means = []
    control_means = []

    yes_sems = []
    no_sems = []
    control_sems = []

    # Extract data for each group
    for key in sorted_keys:
        yes_means.append(bars_dict[key]["Yes"]["votes_mean"])
        no_means.append(bars_dict[key]["No"]["votes_mean"])
        control_means.append(bars_dict[key]["Control"]["votes_mean"])

        yes_sems.append(bars_dict[key]["Yes"]["votes_sem"])
        no_sems.append(bars_dict[key]["No"]["votes_sem"])
        control_sems.append(bars_dict[key]["Control"]["votes_sem"])

    increment = 0.3
    positions = np.arange(0, len(labels) * increment, increment)
    bar_width = 0.05

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        positions - bar_width,
        yes_means,
        bar_width,
        label="Yes",
        yerr=yes_sems,
        capsize=5,
    )
    ax.bar(
        positions,
        control_means,
        bar_width,
        label="Control",
        yerr=control_sems,
        capsize=5,
    )
    ax.bar(positions + bar_width, no_means, bar_width, label="No", yerr=no_sems, capsize=5)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Mean Vote Percentage")
    ax.set_xlabel("(influencer_model, n_interactions)")
    ax.set_title("Mean Vote Percentage by Influencer Model and Number of Interactions")
    ax.legend(title="Influencer target", loc="upper left", bbox_to_anchor=(1, 1))

    # Add horizontal gridlines
    ax.yaxis.grid(True, linestyle="--", which="major", color="grey", alpha=0.25)
    plt.tight_layout()


def _make_success_plot(bars_dict):
    # Sort keys for a consistent ordering in the legend
    sorted_keys = sorted(bars_dict.keys(), key=lambda x: (x[0], x[1]))

    increment = 0.5
    positions = np.arange(0, len(sorted_keys) * increment, increment)
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    # Iterate through each (model, number) pair and plot bars for success
    means = []
    sems = []
    for i, key in enumerate(sorted_keys):
        combined_successes = [
            bars_dict[key][vote]["successes"]
            for vote in bars_dict[key]
            if "successes" in bars_dict[key][vote]
        ]
        # combined_successes is not flat! it's a list of two lists, and I think we should
        # be computing the SEM using the joint length
        flat_combined_successes = [elem for sublist in combined_successes for elem in sublist]
        success_mean = np.mean(flat_combined_successes)
        print(f"{key}: {success_mean = }")
        success_sem = np.std(flat_combined_successes) / np.sqrt(len(flat_combined_successes))
        print(f"{key}: {success_sem = }")
        print(f"{key}: {flat_combined_successes =}")
        print(f"{key}: {len(flat_combined_successes) =}")
        means.append(success_mean)
        sems.append(success_sem)
    ax.bar(positions, means, bar_width, yerr=sems, capsize=5)
    ax.axhline(y=0.5, color="r", linestyle="-")

    ax.set_xticks(positions)
    ax.set_xticklabels(sorted_keys)
    ax.set_ylabel("Combined Success Rate")
    ax.set_xlabel("(influencer_model, n_interactions)")
    ax.set_title("Success rate")

    ax.yaxis.grid(True, linestyle="--", which="major", color="grey", alpha=0.25)
    plt.tight_layout()


if __name__ == "__main__":
    main()
