"""Take results from recent experiments and make a bar plot"""
import argparse
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd

from evals.utils import log_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    log_dir = args.log_dir
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    df = load_mmp_results_from_dir(log_dir)

    # Convert list containing one string to string
    df["con_artist_model"] = [x[0] for x in df["con_artist_model"]]

    # Plot results for all variations
    prompt_type = df["prompt_version"].unique()
    con_artist_models = df["con_artist_model"].unique()
    mark_model = df["mark_model"].unique()
    for i in prompt_type:
        for j in con_artist_models:
            for k in mark_model:
                make_plot(df, out_dir=out_dir, con_artist_model=j, mark_model=k, prompt_type=i)


def load_mmp_results_from_dir(log_dir: Union[str, Path]) -> pd.DataFrame:
    rows = []
    final_results_dict = log_utils.get_final_results_from_dir(log_dir)

    for path, final_results in final_results_dict.items():
        spec = log_utils.extract_spec(path)
        (
            con_artist_model,
            mark_model,
            prompt_version,
            num_experiments,
            turn_cap,
            duration_cap_minutes,
        ) = parse_spec(spec)

        rows.append(
            {
                "con_artist_model": con_artist_model,
                "mark_model": mark_model,
                "prompt_version": prompt_version,
                "num_experiments": num_experiments,
                "turn_cap": turn_cap,
                "duration_cap_minutes": duration_cap_minutes,
                "donation_count": final_results["donation_count"],
                "total_donation_amt": final_results["total_donation_amt"],
            }
        )
    return pd.DataFrame(rows)


def parse_spec(spec: dict) -> tuple[str, bool, int]:
    """parse the spec from a MMP run"""
    con_artist_model = spec["completion_fns"]
    mark_model = spec["run_config"]["eval_spec"]["args"]["mark"]
    prompt_version = spec["run_config"]["eval_spec"]["args"]["prompt_version"]
    num_experiments = spec["run_config"]["eval_spec"]["args"]["num_experiments"]
    turn_cap = spec["run_config"]["eval_spec"]["args"]["turn_cap"]
    duration_cap_minutes = spec["run_config"]["eval_spec"]["args"]["duration_cap_minutes"]

    return (
        con_artist_model,
        mark_model,
        prompt_version,
        num_experiments,
        turn_cap,
        duration_cap_minutes,
    )


def make_plot(df, out_dir, con_artist_model, mark_model, prompt_type):
    variation = df[df["prompt_version"] == prompt_type]
    variation = df[df["con_artist_model"] == con_artist_model]
    xs = variation["turn_cap"]
    dollar_amts = variation["total_donation_amt"]
    donation_frequency = variation["donation_count"]

    # Sort so line is plotted correctly
    results_amts = zip(xs, dollar_amts)
    results_amts = sorted(results_amts, key=lambda x: x[0])

    results_freq = zip(xs, donation_frequency)
    results_freq = sorted(results_freq, key=lambda x: x[0])

    fig, ax1 = plt.subplots()
    fig.suptitle(
        f"Donation frequency and amounts, {con_artist_model} conartist, {prompt_type} {mark_model} mark"
    )

    ax1.set_xlabel("Conversation duration (turns)")
    ax1.set_ylabel("Donation amount (USD)")

    # Plot the left axis data
    ax1.plot(
        [float(x[0]) for x in results_amts],
        [float(x[1]) for x in results_amts],
        color="blue",
        label="Donation amount (USD)",
    )

    # Set the x-axis ticks and labels
    ax1.set_xticks(xs)
    ax1.set_xticklabels(xs)

    # Create a second y-axis on the right side
    ax2 = ax1.twinx()

    # Set the y-axis label for the right axis
    ax2.set_ylabel("Number of donations")

    # Plot the right axis data
    ax2.plot(
        [float(x[0]) for x in results_freq],
        [float(x[1]) for x in results_freq],
        color="red",
        label="Number of donations",
    )

    # Add legend for both axes
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.savefig(out_dir / f"{prompt_type}_duration_donation_frequency_vs_dollar_amts.png")
    plt.show()


if __name__ == "__main__":
    main()
