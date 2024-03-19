import argparse
import copy
import os
import pathlib
from typing import Sequence

import chess.pgn
import requests
import zstandard
from tqdm import tqdm

from evals.elsuite.cant_do_that_anymore.chess.board import BoardController
from evals.elsuite.cant_do_that_anymore.chess.utils import Move, parse_piece
from evals.elsuite.cant_do_that_anymore.utils import (
    assert_boards_consistent,
    dump_sequence_to_jsonl,
    initialise_boards,
)


def prepare_lichess_2014_dataset(out_dir: str) -> str:
    """
    Downloads and extracts Lichess 2014 April dataset, returns the
    path to the extracted .pgn file
    """
    fname = "lichess_db_standard_rated_2014-04.pgn.zst"
    raw_data_out_path = os.path.join(out_dir, fname)
    if not os.path.exists(raw_data_out_path):
        url = "https://database.lichess.org/standard/" + fname
        r = requests.get(url)
        open(raw_data_out_path, "wb").write(r.content)

    out_path = os.path.join(out_dir, "pgn_data.pgn")
    if not os.path.exists(out_path):
        input_file = pathlib.Path(raw_data_out_path)
        with open(input_file, "rb") as compressed:
            decomp = zstandard.ZstdDecompressor()
            with open(out_path, "wb") as destination:
                decomp.copy_stream(compressed, destination)

    return out_path


class MoveFilter:
    def __call__(
        self,
        default_controller: BoardController,
        variant_controller: BoardController,
        move: chess.Move,
        player_id: str,
    ) -> bool:
        raise NotImplementedError()


class SpecialMoveFilter(MoveFilter):
    """
    Filters for moves that are:
    1) Legal under the normal rules of chess
    2) Illegal under the variant's rules (i.e. bishop is moved)
    """

    def __call__(
        self,
        default_controller: BoardController,
        variant_controller: BoardController,
        move: Move,
        player_id: str,
    ) -> bool:
        if not is_move_illegal(default_controller, move, player_id) and is_move_illegal(
            variant_controller, move, player_id
        ):
            return True

        return False


class ControlMoveFilter(MoveFilter):
    """
    Finds positions where solvers should have (almost) equivalent predictions under
    both sets of rules
    Filters for moves that are:
    1) Legal under both the normal and variant's rules of chess
    2) Are on a board containing no bishops
    3) Are on a board where no pawns are close to promoting; neither players
    pawns are in their last three rows
    4) Are on a board with more than four pieces between both players
    """

    def __call__(
        self,
        default_controller: BoardController,
        variant_controller: BoardController,
        move: Move,
        player_id: str,
    ) -> bool:
        if is_move_illegal(default_controller, move, player_id):
            return False
        if is_move_illegal(variant_controller, move, player_id):
            return False

        board_state = default_controller.board.board_state
        num_pieces = 0
        for row_idx in range(8):
            for col_idx in range(8):
                _, piece_id = parse_piece(board_state, row_idx, col_idx)
                if piece_id == 2:
                    return False
                elif piece_id == 0:
                    if player_id == "W" and row_idx <= 2:
                        return False
                    elif player_id == "B" and row_idx >= 5:
                        return False
                elif piece_id != -1:
                    num_pieces += 1

        if num_pieces < 4:
            return False

        return True


def is_move_illegal(controller: BoardController, move: chess.Move, player_id: str) -> bool:
    legal_moves = controller.get_player_legal_moves(player_id)
    if move in legal_moves:
        return False
    return True


def find_specific_moves_in_game(
    game: chess.pgn.Game,
    game_idx: int,
    move_filter: MoveFilter,
    default_controller: BoardController,
    variant_controller: BoardController,
    their_controller: chess.Board,
    filter_if_found_previous: bool,
) -> Sequence[dict]:
    """
    Given a game, finds all moves that satisfy the given filter
    If filter_if_found_previous is True, only finds first move in game that
    satisfies filter
    """
    player_id = "W"
    previous_moves = []
    filtered_moves = []
    for move in game.mainline_moves():
        move = move.uci()

        if move_filter(default_controller, variant_controller, move, player_id):
            filtered_moves.append(
                {
                    "game_idx": game_idx,
                    "previous_moves": copy.deepcopy(previous_moves),
                    "next_filtered_moves": [move],
                    "any_previous_move_found": len(filtered_moves) > 0,
                }
            )
            if filter_if_found_previous:
                break

        # Ensure my implementation is correct
        assert_boards_consistent(default_controller, their_controller, player_id)

        # Update boards
        default_controller.update_board(move)
        their_controller.push_san(move)

        variant_controller.board.board_state = default_controller.board.board_state
        variant_controller.previous_moves = default_controller.previous_moves

        player_id = "B" if player_id == "W" else "W"
        previous_moves.append(move)

    return filtered_moves


def create_dataset_of_specific_moves(
    pgn_path: str,
    move_filter: MoveFilter,
    target_num_examples: int,
    filter_if_found_previous: bool,
    filter_for_unique_previous_moves: bool,
    continuously_save: bool,
    out_path: str,
):
    """
    Iterates over games in dataset and filters move according to the given move_filter
    If filter_for_unique_previous_moves is True, filter to only include moves that have
    unique sets of previous moves
    If continuously_save is True, saves dataset everytime it is updated
    """
    pgn = open(pgn_path)
    dataset = []
    unique_previous_moves = set()

    t_bar = tqdm(total=target_num_examples)
    game_idx = 0
    while True:
        game = chess.pgn.read_game(pgn)
        if game is None:
            break

        default_controller, variant_controller, their_controller = initialise_boards()
        filtered_moves = find_specific_moves_in_game(
            game,
            game_idx,
            move_filter,
            default_controller,
            variant_controller,
            their_controller,
            filter_if_found_previous,
        )

        if filter_for_unique_previous_moves:
            for example in filtered_moves:
                previous_moves = example["previous_moves"]
                if set(previous_moves) not in unique_previous_moves:
                    dataset.append(example)
                    unique_previous_moves.add(frozenset(previous_moves))
                    t_bar.update(1)
                    if continuously_save:
                        dump_sequence_to_jsonl(dataset, out_path)

        elif len(filtered_moves) > 0:
            dataset += filtered_moves
            t_bar.update(len(filtered_moves))
            if continuously_save:
                dump_sequence_to_jsonl(dataset, out_path)

        game_idx += 1
        t_bar.set_description(f"Num games examined: {game_idx}")

        if len(dataset) >= target_num_examples:
            break

    return dataset


def main(args: argparse.Namespace):
    lichess_path = prepare_lichess_2014_dataset(args.out_dir)

    if args.make_special_moves:
        move_filter = SpecialMoveFilter()
        dataset_name = "special_moves_dataset.jsonl"
        out_path = os.path.join(args.out_dir, dataset_name)
        dataset = create_dataset_of_specific_moves(
            lichess_path,
            move_filter,
            target_num_examples=args.n_moves,
            filter_if_found_previous=args.filter_if_found_previous,
            filter_for_unique_previous_moves=args.filter_for_unique_previous_moves,
            continuously_save=args.continuously_save,
            out_path=out_path,
        )
        dump_sequence_to_jsonl(dataset, out_path)

    if args.make_control_moves:
        move_filter = ControlMoveFilter()
        dataset_name = "control_moves_dataset.jsonl"
        out_path = os.path.join(args.out_dir, dataset_name)
        dataset = create_dataset_of_specific_moves(
            lichess_path,
            move_filter,
            target_num_examples=args.n_moves,
            filter_if_found_previous=args.filter_if_found_previous,
            filter_for_unique_previous_moves=args.filter_for_unique_previous_moves,
            continuously_save=args.continuously_save,
            out_path=out_path,
        )
        dump_sequence_to_jsonl(dataset, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--n_moves", type=int, default=5000)
    parser.add_argument(
        "--out_dir", type=str, default="./evals/registry/data/cant_do_that_anymore/"
    )
    parser.add_argument(
        "--make_special_moves",
        action="store_true",
        help="Whether to search and build a dataset of special moves",
        default=False,
    )
    parser.add_argument(
        "--make_control_moves",
        action="store_true",
        help="Whether to search and build a dataset of control moves",
        default=False,
    )
    parser.add_argument(
        "--filter_if_found_previous",
        action="store_true",
        help="Whether to filter out moves that have had previous moves that satisfy the filtering condition.",
        default=False,
    )
    parser.add_argument(
        "--filter_for_unique_previous_moves",
        action="store_true",
        help="Whether to only search for moves with unique previous moves (up to such position at the move)",
        default=False,
    )
    parser.add_argument(
        "--continuously_save",
        action="store_true",
        help="Whether to save the dataset everytime a new example has been found",
        default=False,
    )
    args = parser.parse_args()

    main(args)
