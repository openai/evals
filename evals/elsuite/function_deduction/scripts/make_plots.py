import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from evals.utils import log_utils

palette = {
    "Average Baseline": "blue",
    "Full Knowledge Best": "blue",
    "Full Knowledge Random": "blue",

    "Human": "steelblue",

    "gpt-4-32k": "purple",
    "gpt-4-32k w CoT": "purple",

    "gpt-4-base w Few-shot": "orange",
    "gpt-4-base w CoT and Few-shot": "orange",

    "gpt-3.5-turbo-16k": "green",
    "gpt-3.5-turbo-16k w CoT": "green",

    "gemini-pro": "peru",
    "gemini-pro w CoT": "peru",

    "llama-2-13b-chat": "brown",
    "llama-2-13b-chat w CoT": "brown",

    "llama-2-70b-chat": "maroon",
    "llama-2-70b-chat w CoT": "maroon",

    "mixtral-8x7b-instruct": "grey",
    "mixtral-8x7b-instruct w CoT": "grey",
}

solver_to_name = {
    "function_deduction/full_knowledge_best": "Full Knowledge Best",
    "function_deduction/full_knowledge_random": "Full Knowledge Random",
    "function_deduction/average_baseline": "Average Baseline",

    "human_cli": "Human",

    "gpt-4-32k": "gpt-4-32k",
    "function_deduction/cot/gpt-4-32k": "gpt-4-32k w CoT",

    "function_deduction/gpt-4-base": "gpt-4-base w Few-shot",
    "function_deduction/cot/gpt-4-base": "gpt-4-base w CoT and Few-shot",

    "gpt-3.5-turbo-16k": "gpt-3.5-turbo-16k",
    "function_deduction/cot/gpt-3.5-turbo-16k": "gpt-3.5-turbo-16k w CoT",

    "generation/direct/gemini-pro": "gemini-pro",
    "function_deduction/cot/gemini-pro": "gemini-pro w CoT",

    "generation/direct/llama-2-13b-chat": "llama-2-13b-chat",
    "function_deduction/cot/llama-2-13b-chat": "llama-2-13b-chat w CoT",

    "generation/direct/llama-2-70b-chat": "llama-2-70b-chat",
    "function_deduction/cot/llama-2-70b-chat": "llama-2-70b-chat w CoT",

    "generation/direct/mixtral-8x7b-instruct": "mixtral-8x7b-instruct",
    "function_deduction/cot/mixtral-8x7b-instruct": "mixtral-8x7b-instruct w CoT",
}

rename_columns = {
    "adjusted_avg_rounds": "adjusted_avg_score",
    "sem_adjusted_avg_rounds": "sem_adjusted_avg_score",
}


def extract_final_reports(
    datadir: Path, rename_solvers: dict, rename_columns: dict
) -> pd.DataFrame:
    df_rows = []
    for path, results in sorted(list(log_utils.get_final_results_from_dir(datadir).items())):
        spec = log_utils.extract_spec(path)
        solver_path = spec["completion_fns"][0]
        print("adding report for", solver_path)
        df_rows.append(
            {
                "solver": rename_solvers.get(solver_path, solver_path),
                **{rename_columns.get(k, k): v for k, v in results.items()},
            }
        )
    df = pd.DataFrame(df_rows)
    return df


def make_plot(
    df,
    x_column: str,
    y_column: str,
    x_err_column: str,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
):
    # Avg rounds until success (failure counts as 40)
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        x=x_column,
        y=y_column,
        data=df,
        xerr=df[x_err_column] * 1.96,
        palette=palette,
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="x")
    plt.tight_layout()

    # Expanding the x-axis limit
    x_lim = ax.get_xlim()
    ax.set_xlim([x_lim[0], x_lim[1] * 1.05])  # Increase the upper limit by 5%

    # Annotating each bar with its value
    for p in ax.patches:
        width = p.get_width()
        ax.text(
            width + x_lim[1] * 0.02,  # x position of text
            p.get_y() + p.get_height() / 2,  # y position of text
            "{:.1f}".format(width),  # text to be shown
            va="center",
        )  # vertical alignment

    plt.savefig(out_path)
    return


def make_ask_guess_incorrect_plot(df, out_path: Path):
    # Ask/Guess/Incorrect

    ask_guess_incorrect_data = {
        "solver": df["solver"],
        "Ask": df["avg_ask_rounds"],
        "SEM Average Ask Rounds": df["sem_avg_ask_rounds"],
        "Guess": df["avg_guess_rounds"],
        "SEM Average Guess Rounds": df["sem_avg_guess_rounds"],
        "Incorrect Format": df["avg_incorrect_format_rounds"],
        "SEM Average Incorrect Format Rounds": df["sem_avg_incorrect_format_rounds"],
    }

    agi_palette = {
        "Ask": "blue",
        "Guess": "pink",
        "Incorrect Format": "red",
    }

    ask_guess_incorrect_df = pd.DataFrame(ask_guess_incorrect_data)

    # Melting the DataFrame to make it suitable for seaborn's factorplot
    melted_df = pd.melt(
        ask_guess_incorrect_df,
        id_vars="solver",
        value_vars=["Ask", "Guess", "Incorrect Format"],
        var_name="Round Type",
        value_name="Average Rounds",
    )

    # Generating the plot for Average Ask/Guess/Incorrect Format Rounds
    plt.figure(figsize=(14, 14))
    ax = sns.barplot(
        x="Average Rounds", y="solver", hue="Round Type", data=melted_df, palette=agi_palette
    )

    plt.xlabel("Average Number of Rounds")
    plt.ylabel("Solver")
    plt.title("Distribution of Type of Responses by Model")
    plt.grid(axis="x")
    plt.legend(title="Response Type")
    plt.tight_layout()

    # Expanding the x-axis limit
    x_lim = ax.get_xlim()
    ax.set_xlim([x_lim[0], x_lim[1] * 1.05])  # Increase the upper limit by 5%

    # Annotating each bar with its value
    for p in ax.patches:
        width = p.get_width()
        ax.text(
            width + 0.1,  # x position of text
            p.get_y() + p.get_height() / 2,  # y position of text
            "{:.1f}".format(width),  # text to be shown
            va="center",
        )  # vertical alignment

    plt.savefig(out_path)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", "-d", type=str, required=True)
    parser.add_argument("--out-dir", "-o", type=str, default="./outputs")
    args = parser.parse_args()
    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)

    df = extract_final_reports(log_dir, solver_to_name, rename_columns)

    # Drop all columns named "complexity*"
    df = df[df.columns.drop(list(df.filter(regex="complexity")))]

    # Creating a new DataFrame with the desired order
    ordered_df = df.set_index("solver").loc[list(solver_to_name.values())].reset_index()
    print(ordered_df)

    make_plot(
        df=ordered_df,
        x_column="adjusted_avg_score",
        y_column="solver",
        x_err_column="sem_adjusted_avg_score",
        title="Adjusted Average Score (Lower is Better)",
        xlabel="Adjusted Average Score",
        ylabel="Solver",
        out_path=out_dir / "avg_adjusted_score.png",
    )

    ordered_df["solved_ratio"] = 100 * ordered_df["solved_ratio"]
    ordered_df["sem_solved_ratio"] = 100 * ordered_df["sem_solved_ratio"]
    make_plot(
        df=ordered_df,
        x_column="solved_ratio",
        y_column="solver",
        x_err_column="sem_solved_ratio",
        title="Solved Samples Ratio (Higher is Better)",
        xlabel="Solved Ratio (%)",
        ylabel="Solver",
        out_path=out_dir / "solved_ratio.png",
    )

    make_plot(
        df=ordered_df,
        x_column="avg_success_rounds",
        y_column="solver",
        x_err_column="sem_avg_success_rounds",
        title="Average Number of Rounds for Solved Samples (Lower is Better)",
        xlabel="No. of Rounds",
        ylabel="Solver",
        out_path=out_dir / "avg_success_rounds.png",
    )

    make_ask_guess_incorrect_plot(
        df=ordered_df,
        out_path=out_dir / "ask_guess_incorrect.png",
    )


if __name__ == "__main__":
    main()
