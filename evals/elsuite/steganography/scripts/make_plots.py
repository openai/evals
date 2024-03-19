import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from evals.utils import log_utils


def extract_results(datadir: Path) -> pd.DataFrame:
    df_rows = []
    for path, results in log_utils.get_final_results_from_dir(datadir).items():
        spec = log_utils.extract_spec(path)
        model = spec["completion_fns"][0]
        prompt_version = spec["run_config"]["eval_spec"]["args"]["prompt_version"]
        df_rows.append(
            {
                "model": model,
                "prompt_version": prompt_version,
                **results,
            }
        )
    df = pd.DataFrame(df_rows)
    return df


def make_undetected_acc_plot(df: pd.DataFrame, outpath: Path):
    conv_to_percentage = 100
    # Plot total accuracy of undetected payloads for groups of (model, prompt_version)
    means = df.groupby(["model", "prompt_version"])["undetected_acc"].mean().reset_index()
    means["undetected_acc"] = means["undetected_acc"] * conv_to_percentage
    plt.figure()
    plt.grid(axis="y")
    sns.barplot(x="prompt_version", y="undetected_acc", hue="model", data=means)
    plt.ylabel("Mean Undetected Accuracy ↑ (%)")
    plt.xlabel("Prompt Version")
    plt.savefig(outpath, bbox_inches="tight")
    plt.show()


def make_reconstruction_vs_detectability_plot(df: pd.DataFrame, outpath: Path):
    conv_to_percentage = 100
    # Plot semantic distance VS detectability for groups of (model, prompt_version)
    anno, type_label, xx, yy = [], [], [], []
    plt.figure()
    plt.grid()
    for (model, prompt_version), group in sorted(df.groupby(["model", "prompt_version"])):
        detectability = group["detectability"].mean() * conv_to_percentage
        semantic_distance = group["semantic_distance"].mean() * conv_to_percentage
        print(
            f"model={model}, prompt_version={prompt_version}, detectability={detectability}, semantic_distance={semantic_distance}"
        )
        anno.append(f"{prompt_version}")
        type_label.append(model)
        xx.append(detectability)
        yy.append(semantic_distance)
    # Plot
    for label in set(type_label):
        plt.scatter(
            [x for x, t in zip(xx, type_label) if t == label],
            [y for y, t in zip(yy, type_label) if t == label],
            label=label,
        )
    plt.legend()
    for ann, x, y in zip(anno, xx, yy):
        plt.annotate(ann, (x, y))
    plt.xlabel("Detectability ↓ (%)")
    plt.ylabel("Semantic Distance ↓ (%)")
    # Axis limits 0-100
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.savefig(outpath, bbox_inches="tight")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", "-d", type=str, required=True)
    parser.add_argument("--out_dir", "-o", type=str, default="./outputs")
    args = parser.parse_args()
    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)

    df = extract_results(log_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    make_undetected_acc_plot(df, out_dir / "undetected_acc.png")
    make_reconstruction_vs_detectability_plot(df, out_dir / "reconstruction_vs_detectability.png")


if __name__ == "__main__":
    main()
