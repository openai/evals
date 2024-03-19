from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from evals.elsuite.identifying_variables.metrics import compute_metric_posthoc
from evals.elsuite.identifying_variables.scripts.plotting_utils import (
    plot_difficulty_bars,
    plot_solver_bars,
)
from evals.elsuite.identifying_variables.scripts.table_utils import (
    make_main_metric_table,
)
from evals.utils import log_utils

NUM_REPEATS = 3
MAIN_METRICS = [
    "ctrl_nDCG",
    "ctrl_recall",
    "hyp_valid_acc",
    "ind_acc",
    "dep_acc",
    "violation_rate",
]

SOLVERS = [
    "generation/direct/gpt-3.5-turbo",
    "generation/cot/gpt-3.5-turbo",
    "generation/hhh/gpt-4-base",
    "generation/cot_hhh/gpt-4-base",
    "generation/direct/gpt-4-1106-preview",
    "generation/cot/gpt-4-1106-preview",
    "generation/cot/mixtral-8x7b-instruct",
    "generation/cot/llama-2-70b-chat",
    "generation/cot/gemini-pro",
    "identifying_variables/random",
    "identifying_variables/noctrl",
]


RENDERERS = [
    "markdown",
    "csv",
    "json",
    "language-tabular",
    "language-corrset",
    "corrset",
]


def initialize_default_results_dict():
    results_dict = {
        metric: {
            stat: {
                solver: {
                    renderer: {
                        "with tree": ([] if stat == "raw" else 0),
                        "without tree": ([] if stat == "raw" else 0),
                    }
                    for renderer in RENDERERS
                }
                for solver in SOLVERS
            }
            for stat in ["raw", "mean", "sem"]
        }
        for metric in MAIN_METRICS
    }
    return results_dict


def handle_cot_double_sampling(sampling_entries, solver):
    if "cot" in solver:
        sampling_entries = [
            entry
            for entry in sampling_entries
            if (
                # for chat models we filter like this
                isinstance(entry["prompt"], list)
                and entry["prompt"][-1]["content"].startswith(
                    "Given the above reasoning"
                )
                or (
                    # for base models we need to filter like this
                    isinstance(entry["prompt"], str)
                    and "Given the above reasoning" in entry["prompt"]
                )
            )
        ]
    return sampling_entries


def handle_posthoc_metrics(final_results: Dict, log_path: Path, solver: str):
    """
    Computes and includes missing metrics from log file if they are not present
    """
    metric_entries = log_utils.extract_individual_results(log_path)
    sampling_entries = log_utils.extract_individual_results(log_path, "sampling")
    # filter out cot double samplings
    sampling_entries = handle_cot_double_sampling(sampling_entries, solver)
    # this is necessary because we originally didnt compute recall in the eval
    for metric in MAIN_METRICS:
        if metric not in final_results.keys():
            final_results[metric] = compute_metric_posthoc(
                metric, metric_entries, sampling_entries
            )

    return final_results


def populate_default_results_dict(results_dict, results_dir):
    for log in tqdm(results_dir.glob("*.log"), total=222):
        spec = log_utils.extract_spec(log)
        solver = spec["completion_fns"][0]
        run_config = spec["run_config"]
        renderer = run_config["eval_spec"]["args"]["renderer"]
        show_tree = "show_tree=True" in run_config["command"]
        tree_key = "with tree" if show_tree else "without tree"
        if renderer not in RENDERERS and solver != "identifying_variables/random":
            continue
        if solver not in SOLVERS:
            continue

        final_results = log_utils.extract_final_results(log)
        final_results = handle_posthoc_metrics(final_results, log, solver)

        for metric, value in final_results.items():
            if metric in MAIN_METRICS:
                results_dict[metric]["raw"][solver][renderer][tree_key].append(value)
                raw = results_dict[metric]["raw"][solver][renderer][tree_key]
                results_dict[metric]["mean"][solver][renderer][tree_key] = np.mean(raw)
                results_dict[metric]["sem"][solver][renderer][tree_key] = np.std(
                    raw
                ) / np.sqrt(NUM_REPEATS)
    for metric in results_dict.keys():
        del results_dict[metric]["raw"]
    return results_dict


def make_default_tables(results_dict: Dict, save_dir: Path):
    for metric in tqdm(MAIN_METRICS):
        make_main_metric_table(results_dict, metric, SOLVERS, RENDERERS, save_dir)


def extract_default_results_dict(results_dir: Path):
    results_dict = initialize_default_results_dict()
    results_dict = populate_default_results_dict(results_dict, results_dir)

    return results_dict


def make_default_plots(results_dict: Dict, save_dir: Path):
    all_solvers = list(results_dict["ctrl_nDCG"]["mean"].keys())
    bar_solvers, baseline_solvers = all_solvers[:-2], all_solvers[-2:]

    metrics = ["ctrl_nDCG", "ctrl_recall"]
    metric_labels = ["Control Variable Retrieval nDCG*", "Control Variable Recall"]
    fig_heights = [6, 5]

    for metric, metric_label, fig_height in tqdm(
        zip(metrics, metric_labels, fig_heights)
    ):
        plot_solver_bars(
            bar_solvers,
            baseline_solvers,
            results_dict[metric],
            metric_label,
            fig_height,
            save_dir / f"{metric}.png",
        )


def extract_large_results_dict(results_dir: Path) -> Dict:
    ctrl_nDCG_bins = list(range(0, 9))
    results_dict = {}
    for log in tqdm(results_dir.glob("*.log"), total=12):
        spec = log_utils.extract_spec(log)
        final_results = log_utils.extract_final_results(log)
        solver = spec["completion_fns"][0]
        renderer = spec["split"]
        key = f"{solver};{renderer}"
        if key not in results_dict:
            results_dict[key] = {
                bbin: {"raw": [], "mean": None, "sem": None} for bbin in ctrl_nDCG_bins
            }

        for bbin in ctrl_nDCG_bins:
            results_dict[key][bbin]["raw"].append(
                final_results[f"ctrl_nDCG-n_ctrl_vars-{bbin}"]
            )
    for key in results_dict.keys():
        for bbin in ctrl_nDCG_bins:
            mean = np.mean(results_dict[key][bbin]["raw"])
            sem = np.std(results_dict[key][bbin]["raw"]) / 3
            results_dict[key][bbin]["mean"] = mean
            results_dict[key][bbin]["sem"] = sem
            del results_dict[key][bbin]["raw"]

    return results_dict


def make_large_plot(large_results_dir: Dict, save_dir: Path):
    ctrl_vars_bins = list(range(0, 9))
    plot_difficulty_bars(
        large_results_dir, ctrl_vars_bins, save_dir / "ctrl_nDCG_difficulty.png"
    )


def np_nan_if_none(input_num):
    if input_num is None:
        return np.nan
    else:
        return input_num


def zero_if_none(input_num):
    if input_num is None:
        return 0
    else:
        return input_num


def round_if_not_nan(input_num):
    if np.isnan(input_num):
        return input_num
    else:
        return round(input_num)


def make_token_per_sample_df(solver_to_eval, solver_to_tokens) -> pd.DataFrame:
    tokens_per_sample_df = pd.DataFrame(
        index=solver_to_eval.keys(),
        columns=[
            "input tokens/sample",
            "output tokens/sample",
            "total tokens/sample",
        ],
    )
    for solver in solver_to_tokens.keys():
        # print(solver_to_tokens[solver])
        input_mean = np.nanmean(solver_to_tokens[solver]["input"])
        output_mean = np.nanmean(solver_to_tokens[solver]["output"])
        total_mean = np.nanmean(solver_to_tokens[solver]["total"])
        # print([input_mean, output_mean, total_mean])
        tokens_per_sample_df.loc[solver] = [
            round_if_not_nan(input_mean),
            round_if_not_nan(output_mean),
            round_if_not_nan(total_mean),
        ]
    solver_to_index = {
        "generation/hhh/gpt-4-base": "HHH GPT-4-base (corrset, no tree)",
        "generation/direct/gpt-3.5-turbo": "Direct GPT-3.5-turbo (corrset, no tree)",
        "generation/direct/gpt-4-1106-preview": "Direct GPT-4-1106-preview (corrset, no tree)",
        "generation/cot_hhh/gpt-4-base": "CoT HHH GPT-4-base (language-tabular, with tree)",
        "generation/cot/gpt-3.5-turbo": "CoT GPT-3.5-turbo (language-tabular, with tree)",
        "generation/cot/gpt-4-1106-preview": "CoT GPT-4-1106-preview (language-tabular, with tree)",
    }
    tokens_per_sample_df = tokens_per_sample_df.rename(index=solver_to_index)
    return tokens_per_sample_df


def count_tokens(results_dir: Path, total) -> Tuple[Dict, pd.DataFrame]:
    eval_names = [
        "identifying_variables.corrset.default",
        "identifying_variables.language-tabular.default",
    ]
    solver_names = [
        "generation/hhh/gpt-4-base",
        "generation/direct/gpt-3.5-turbo",
        "generation/direct/gpt-4-1106-preview",
        "generation/cot_hhh/gpt-4-base",
        "generation/cot/gpt-3.5-turbo",
        "generation/cot/gpt-4-1106-preview",
    ]
    solver_to_eval = {
        solver: eval_names[0] if "cot" not in solver else eval_names[1]
        for solver in solver_names
    }
    solver_to_tree = {
        solver: False if "cot" not in solver else True for solver in solver_names
    }
    solver_to_tokens = {
        solver: {"input": [], "output": [], "total": []} for solver in solver_names
    }
    total_input = 0
    total_output = 0
    for log in tqdm(results_dir.glob("*.log"), total=total):
        spec = log_utils.extract_spec(log)
        solver = spec["completion_fns"][0]
        if solver not in solver_names:
            print(f"Skipping {solver}: token counting not supported.")
            continue
        eval_name = spec["eval_name"]
        seed = spec["run_config"]["seed"]
        tree = "show_tree=True" in spec["run_config"]["command"]
        samplings = log_utils.extract_individual_results(log, "sampling")
        samplings = handle_cot_double_sampling(samplings, solver)
        for sampling in samplings:
            usage = sampling["usage"]
            if (
                solver in solver_to_eval
                and eval_name == solver_to_eval[solver]
                and seed == 1
                and tree != solver_to_tree[solver]
            ):
                solver_to_tokens[solver]["input"].append(
                    np_nan_if_none(usage["prompt_tokens"])
                )
                solver_to_tokens[solver]["output"].append(
                    np_nan_if_none(usage["completion_tokens"])
                )
                solver_to_tokens[solver]["total"].append(
                    np_nan_if_none(usage["total_tokens"])
                )
            total_input += zero_if_none(usage["prompt_tokens"])
            total_output += zero_if_none(usage["completion_tokens"])

    total_tokens = {"input": total_input, "output": total_output}
    tokens_per_sample_df = make_token_per_sample_df(solver_to_eval, solver_to_tokens)

    return total_tokens, tokens_per_sample_df


def make_total_tokens_table(default_total: Dict, large_total: Dict) -> pd.DataFrame:
    """
    Makes a dataframe where the index is "default" "large" and the columns are
    "input", "output"; showing the total number of input and output tokens for
    our experiments on each dataset.
    """
    total_tokens_df = pd.DataFrame(
        {
            "input": [default_total["input"], large_total["input"]],
            "output": [default_total["output"], large_total["output"]],
        },
        index=["default", "large"],
    )
    return total_tokens_df


def make_token_count_tables(
    default_results_dir: Path, large_results_dir: Path, save_dir: Path
):
    default_total_tokens, default_per_sample_tokens_df = count_tokens(
        default_results_dir, total=222
    )
    large_total_tokens, _ = count_tokens(large_results_dir, total=12)

    total_tokens_df = make_total_tokens_table(default_total_tokens, large_total_tokens)

    # save the tables
    total_tokens_df.to_csv(save_dir / "total_tokens.csv")
    default_per_sample_tokens_df.to_csv(save_dir / "per_sample_tokens.csv")


def main(default_results_dir: Path, large_results_dir: Path, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Parsing default dataset results...")
    default_results_dict = extract_default_results_dict(default_results_dir)
    print("Making default dataset tables...")
    make_default_tables(default_results_dict, save_dir)
    print("Making default dataset plots...")
    make_default_plots(default_results_dict, save_dir)

    print("Parsing large dataset results...")
    large_results_dict = extract_large_results_dict(large_results_dir)
    print("Making large dataset plot...")
    make_large_plot(large_results_dict, save_dir)

    print("Making token count tables...")
    make_token_count_tables(default_results_dir, large_results_dir, save_dir)
    print("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process results")
    parser.add_argument(
        "--default_results_dir",
        type=str,
        help="Path to directory containing .log files from experiments on default dataset",
    )
    parser.add_argument(
        "--large_results_dir",
        type=str,
        help="Path to directory containing .log files from experiments on large dataset",
    )
    parser.add_argument(
        "--save_dir", type=str, help="Path to directory to save plots and tables to"
    )

    args = parser.parse_args()

    main(
        Path(args.default_results_dir),
        Path(args.large_results_dir),
        Path(args.save_dir),
    )
