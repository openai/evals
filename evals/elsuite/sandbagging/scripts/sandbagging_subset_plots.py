import argparse
from pathlib import Path

from utils import compute_all_model_scores, extract_results, make_sandbagging_plot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", "-d", type=str, required=True)
    parser.add_argument("--out_dir", "-o", type=str, default="./outputs")
    args = parser.parse_args()
    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    df = extract_results(log_dir)
    colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]

    print("Results for sandbagging on MMLU astronomy questions:")
    model_log = make_sandbagging_plot(
        df,
        out_dir=out_dir,
        file_name="sb_subset.png",
        title="Sandbagging on MMLU astronomy questions",
        colors=colors,
        baseline_eval_name="no-sandbagging-subset",
        sb_eval_name="sandbagging-subset",
    )
    scores = compute_all_model_scores(model_log, thresh_targets=True)
    for name, s in scores.items():
        print(f"{name} achieved a score of: {round(s, 2)}")

    print("Results for performance on MMLU normal questions when sandbagging:")
    model_log = make_sandbagging_plot(
        df,
        out_dir=out_dir,
        file_name="sb_subset_normal.png",
        title="Performance on MMLU normal questions when sandbagging",
        colors=colors,
        baseline_eval_name="no-sandbagging-all",
        sb_eval_name="sandbagging-all",
        plot_all_data_together=True,
    )
    # Change targets to be each models baseline
    for name, v in model_log.items():
        baseline = v["baseline"]
        no_targets = len(v["targets"])
        v["targets"] = [baseline] * no_targets

    scores = compute_all_model_scores(model_log, thresh_targets=False)
    for name, s in scores.items():
        print(f"{name} achieved a score of: {round(s, 2)}")


if __name__ == "__main__":
    main()
