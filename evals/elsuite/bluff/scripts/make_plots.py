import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from evals.utils import log_utils


def extract_results(datadir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    main_data = []
    round_data = []
    round_labels = []
    player_0_names = set()
    for _, results in sorted(log_utils.get_final_results_from_dir(datadir).items()):
        num_rounds = results["valid_samples"] * len(results["player_0_per_round_wins"])

        #   We don't need the "strategic_" prefix
        player_0 = (
            results["player_0"][10:]
            if results["player_0"].startswith("strategic_")
            else results["player_0"]
        )
        player_1 = (
            results["player_1"][10:]
            if results["player_1"].startswith("strategic_")
            else results["player_1"]
        )

        main_data.append([player_0, player_1, results["player_0_win_ratio"], num_rounds])
        round_labels.append(player_0 + " vs " + player_1)
        round_data.append(
            [wins / results["valid_samples"] for wins in results["player_0_per_round_wins"]]
        )
        player_0_names.add(player_0)

    #   We want to have the same palette for both plots, so we create it here
    model_color_map = {name: color for name, color in zip(player_0_names, sns.color_palette())}

    df_main = pd.DataFrame(
        main_data, columns=["player_0", "player_1", "player_0_win_ratio", "num_rounds"]
    )
    df_round = pd.DataFrame(round_data, round_labels).T

    return df_main, df_round, model_color_map


def make_main_metric_plots(df: pd.DataFrame, palette: dict, outdir: Path) -> None:
    sns.set_theme(style="darkgrid")

    opponents = df["player_1"].unique()
    for opponent in opponents:
        opp_df = df[df["player_1"] == opponent].reset_index()
        outpath = outdir / f"main_{opponent}.png"
        _make_main_metric_plot(opp_df, palette, opponent, outpath)


def _make_main_metric_plot(df: pd.DataFrame, palette: dict, opponent: str, outpath: Path) -> None:
    #   Calculate error bars
    error_bars = {}
    for ix, row in df.iterrows():
        winrate = row["player_0_win_ratio"]
        # standard error of the mean (SEM) for binary variables
        # sqrt(p * (1 - p) / n)
        sem = (winrate * (1 - winrate) / row["num_rounds"]) ** 0.5
        error_bars[ix] = (winrate - 2 * sem, winrate + 2 * sem)

    #   Duplicate the rows so that `errorbar=func` argument in `sns.catplot` works.
    #   This is a super-ugly fix, but is probably "cleanest" from the POV of the
    #   amount of code written.
    duplicated_rows_df = pd.concat([df, df], ignore_index=True)

    g = sns.catplot(
        data=duplicated_rows_df,
        kind="bar",
        legend=False,
        x="player_0",
        y="player_0_win_ratio",
        errorbar=lambda x: error_bars[x.index[0]],
        errwidth=1,
        capsize=0.1,
        palette=palette,
        aspect=1.5,
    )
    g.set(ylim=(0, 1))
    g.despine(left=True)
    g.set(title=f"Win ratio against {opponent}")
    g.set(xlabel=None, ylabel="% of rounds won")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.show()


def make_per_round_plots(df: pd.DataFrame, palette: dict, outdir: Path) -> None:
    sns.set_theme(style="darkgrid")

    opponents = set(col.split(" vs ")[1] for col in df.columns)
    for opponent in opponents:
        opp_df = df[[col for col in df.columns if col.endswith(f" vs {opponent}")]]
        opp_df.columns = [col.split(" vs ")[0] for col in opp_df.columns]
        outpath = outdir / f"per_round_{opponent}.png"
        _make_per_round_plot(opp_df, palette, opponent, outpath)


def _make_per_round_plot(df: pd.DataFrame, palette: dict, opponent: str, outpath: Path) -> None:
    # Sort columns based on their score at round 9
    sorted_columns = df.loc[9].sort_values(ascending=False).index

    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    for col in sorted_columns:
        color = palette[col]
        sns.lineplot(x=df.index, y=df[col], ax=ax, label=col, color=color, linestyle="-")

    ax.set_ylim(0, 1)
    ax.set(xlabel="round number", ylabel="% of rounds won")
    ax.set(title=f"Per-round win ratio against {opponent}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", "-d", type=str, required=True)
    parser.add_argument("--out-dir", "-o", type=str, default="./outputs")
    args = parser.parse_args()
    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)

    df_main, df_round, model_color_map = extract_results(log_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    make_main_metric_plots(df_main, model_color_map, out_dir)
    make_per_round_plots(df_round, model_color_map, out_dir)


if __name__ == "__main__":
    main()
