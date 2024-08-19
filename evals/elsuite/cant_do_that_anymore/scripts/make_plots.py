import argparse
import os
from pathlib import Path
from typing import Sequence

import pandas as pd
from matplotlib import pyplot as plt

from evals.elsuite.cant_do_that_anymore.chess.utils import parse_piece
from evals.elsuite.cant_do_that_anymore.utils import initialise_boards
from evals.utils.log_utils import (
    extract_individual_results,
    extract_spec,
    get_final_results_from_dir,
)


def extract_results(datadir: Path) -> pd.DataFrame:
    df_agg = []  # Aggregated results
    df_samples = []  # Per sample results
    for path, results in sorted(list(get_final_results_from_dir(datadir).items())):
        spec = extract_spec(path)
        solver_path = Path(spec["completion_fns"][0])
        model = solver_path.name
        solver = solver_path.parent.name
        # Remove root section of path, which is the eval name
        solver_path = solver_path.relative_to(solver_path.parts[0])
        # Aggregated
        df_agg.append(
            {
                "solver_path": str(solver_path),
                "model": str(model),
                "solver": str(solver),
                **spec["run_config"]["eval_spec"]["args"],
                **results,
            }
        )
        # Per-sample
        for res in extract_individual_results(path):
            df_samples.append(
                {
                    "solver_path": str(solver_path),
                    "model": str(model),
                    "solver": str(solver),
                    **spec["run_config"]["eval_spec"]["args"],
                    **res,
                }
            )
    df_agg = pd.DataFrame(df_agg)
    df_samples = pd.DataFrame(df_samples)
    return df_agg, df_samples


def render_results(df: pd.DataFrame, out_dir: Path):
    agg_operations = {
        "predicted_move_proportion": ["mean", "sem"],
        "predicted_move_in_variant_proportion": ["mean", "sem"],
    }
    df = df.groupby("solver_path").agg(agg_operations).reset_index()
    df = df.round(2)
    print(df.to_csv(index=False))
    df.to_csv(os.path.join(out_dir, "results.csv"), index=False)


def compute_num_previous_bishop_moves(previous_moves: Sequence[str]) -> int:
    controller, _, _ = initialise_boards()

    num_previous_bishop_moves = 0
    for move in previous_moves:
        start_coord = controller.notation_parser._str_to_move(
            move, controller.board.board_state
        ).start_coord
        _, piece_id = parse_piece(controller.board.board_state, start_coord[0], start_coord[1])
        if piece_id == 2:
            num_previous_bishop_moves += 1

        controller.update_board(move)

    return num_previous_bishop_moves


def plot_diagonal_bishop_results(df: pd.DataFrame, out_dir: Path):
    # Get number of previous bishop moves
    df["num_previous_bishop_moves"] = [
        compute_num_previous_bishop_moves(i) for i in df["previous_moves"]
    ]

    # Calculate headline metrics per solver, and number of previous moves
    agg_operations = {
        "predicted_move_in_variant": ["mean"],
    }
    df = df.groupby(["solver_path", "num_previous_bishop_moves"]).agg(agg_operations).reset_index()

    # Plot separately for each solver
    for model, group in df.groupby("solver_path"):
        plt.plot(
            group["num_previous_bishop_moves"],
            group["predicted_move_in_variant"],
            label=model,
        )

    plt.xlabel("Num previous bishop moves")
    plt.ylabel("Proportion of (illegal) predicted diagonal bishop moves")
    plt.ylim([0, 1])
    plt.legend()
    plt.savefig(os.path.join(out_dir, "diagonal.png"))
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", "-d", type=str, required=True)
    parser.add_argument("--out_dir", "-o", type=str, required=True)
    parser.add_argument("--diagonal_variant", action="store_true", default=False)
    args = parser.parse_args()
    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    df_agg, df_samples = extract_results(log_dir)
    render_results(df_agg, out_dir)

    if args.diagonal_variant:
        plot_diagonal_bishop_results(df_samples, out_dir)


if __name__ == "__main__":
    main()
