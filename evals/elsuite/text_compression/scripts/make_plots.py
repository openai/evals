import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

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


def make_reconstruction_vs_compression_plot(df: pd.DataFrame, outpath: Path):
    reconstruction_metric = "character_error_rate_cap1"
    compression_metric = "compression_ratio_cap1"
    # Calculate metrics for groups of (model, prompt_version)
    conv2percentage = 100
    anno, type_label, xx, yy = [], [], [], []
    for (model, prompt_version), group in sorted(df.groupby(["model", "prompt_version"])):
        compression_ratio = group[compression_metric].mean() * conv2percentage
        character_error_rate = group[reconstruction_metric].mean() * conv2percentage
        print(
            f"model={model}, prompt_version={prompt_version}, compression_ratio={compression_ratio}, reconstruction_metric={character_error_rate}"
        )
        anno.append(prompt_version)
        type_label.append(model)
        xx.append(compression_ratio)
        yy.append(character_error_rate)
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
    plt.xlabel("Compression Ratio ↓ (% of original text)")
    plt.ylabel("Character Error Rate ↓ (% of original text)")
    # Axis limits 0-100
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    # Add grid
    plt.grid(linestyle="-", alpha=0.5)
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
    make_reconstruction_vs_compression_plot(df, out_dir / "reconstruction_vs_compression.png")


if __name__ == "__main__":
    main()
