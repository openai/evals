import argparse
import copy
import os
import random
from typing import Optional, Sequence

from stockfish import Stockfish
from tqdm import tqdm

from evals.elsuite.cant_do_that_anymore.chess.board import BoardController
from evals.elsuite.cant_do_that_anymore.chess.move_variants import DIAGONAL_MOVES
from evals.elsuite.cant_do_that_anymore.chess.utils import (
    Move,
    coord_within_board,
    move_crosses_pieces,
    parse_piece,
)
from evals.elsuite.cant_do_that_anymore.utils import dump_sequence_to_jsonl, initialise_boards

# NOTE change threads, hash depending on hardware
# https://pypi.org/project/stockfish/
STOCKFIAH_MOVES_CONSIDERED = 5
STOCKFISH_DEPTH = 18
STOCKFISH_PARAMS = {
    "Debug Log File": "",
    "Contempt": 0,
    "Min Split Depth": 0,
    "Threads": 8,
    "Ponder": "false",
    "Hash": 4096,
    "MultiPV": 1,
    "Skill Level": 10,
    "Move Overhead": 10,
    "Minimum Thinking Time": 20,
    "Slow Mover": 100,
    "UCI_Chess960": "true",
    "UCI_LimitStrength": "false",
    "UCI_Elo": 1500,
}


def get_stockfish_move(stockfish: Stockfish, num_moves_to_consider: int) -> str:
    """
    Gets the next move predicted by stockfish. Gets top n predictions and
    selects randomly weighted by each move's centipawn value
    Filters out bishop promotions, since our variant shouldn't have bishops
    """
    # Get top moves, filter out bad ones
    top_moves = stockfish.get_top_moves(num_moves_to_consider)

    # Filter out bishop promotions
    top_moves = [i for i in top_moves if not i["Move"].endswith("b")]

    # If stockfish considers moves that it knows will lead to mate, only
    # select from these moves
    mates = [i for i in top_moves if i["Mate"] is not None]
    if len(mates) > 0:
        top_moves = mates

    # Ensures centipawn value isn't None
    if all([i["Centipawn"] is None for i in top_moves]):
        for move in top_moves:
            move["Centipawn"] = 1
    else:
        top_moves = [i for i in top_moves if i["Centipawn"] is not None]

    # Makes all centipawns positive
    min_centipawn_value = min([i["Centipawn"] for i in top_moves])
    for move in top_moves:
        move["Centipawn"] += abs(min_centipawn_value)

    # Normalise centipawn to a probability distribution
    centipawn_sum = sum([i["Centipawn"] for i in top_moves])
    for move in top_moves:
        move["prob"] = move["Centipawn"] / centipawn_sum

    # Pick move randomly
    prob = random.uniform(0, 1)
    selected_move = None
    for move in top_moves:
        prob -= move["prob"]
        if prob <= 0:
            selected_move = move["Move"]
            break

    return selected_move


def parse_stockfish_move(controller: BoardController, move: str) -> str:
    """
    When stockfish outputs a castling move, the move is from the kings position to the
    rooks position, e.g. "e8a8"
    In my framework castling is indicated by the start+end position of the king, e.g. "e8c8"
    This functions converts the stockfish notation to my notation
    """
    move = controller.notation_parser._str_to_move(move, controller.board.board_state)
    _, piece_id = parse_piece(
        controller.board.board_state, move.start_coord[0], move.start_coord[1]
    )

    # If castling move
    dy = move.target_coord[1] - move.start_coord[1]
    if piece_id == 5:
        if dy > 2 or dy < -2:
            direction = dy / abs(dy)
            if direction == 1:  # Kingside castling
                move.target_coord = [move.target_coord[0], move.target_coord[1] - 1]
            else:  # Queenside castling
                move.target_coord = [move.target_coord[0], move.target_coord[1] + 2]

    move = controller.notation_parser._move_to_str(move, controller.board.board_state)
    return move


def get_bishop_diagonal_moves(controller: BoardController, player_id: str) -> Sequence[str]:
    """
    Gets all possible diagonal moves that a bishop could make on a board, even if the bishop isn't
    allowed to move diagonally under the board's rules
    """
    # Find all bishops on board
    bishop_coords = []
    board_state = controller.board.board_state
    for row_idx in range(8):
        for col_idx in range(8):
            piece_color, piece_id = parse_piece(board_state, row_idx, col_idx)
            if piece_color == player_id and piece_id == 2:
                bishop_coords.append([row_idx, col_idx])

    # Find all possible diagonal movements of each bishop
    bishop_diagonal_moves = []
    for row_idx, col_idx in bishop_coords:
        for transformation in DIAGONAL_MOVES:
            new_coord = [row_idx + transformation[0], col_idx + transformation[1]]
            move = Move([row_idx, col_idx], new_coord, promotion=None, castling=False)

            # If piece doesn't move
            if transformation[0] == 0 and transformation[1] == 0:
                continue
            # If transformation moves piece outside board
            if not coord_within_board(new_coord[0], new_coord[1]):
                continue
            # If transformation moves onto piece of same color
            piece_color, _ = parse_piece(controller.board.board_state, new_coord[0], new_coord[1])
            if piece_color == player_id:
                continue
            # If move crosses friendly pieces
            if move_crosses_pieces(controller.board.board_state, move):
                continue

            move = controller.notation_parser._move_to_str(move, controller.board.board_state)
            bishop_diagonal_moves.append(move)

    return bishop_diagonal_moves


def find_specific_moves_in_game(
    game_idx: int,
    variant_controller: BoardController,
    filter_if_found_previous: bool,
    max_moves: int,
) -> Sequence[dict]:
    """
    Simulates an individual game, using the variant's rules. Finds all possible
    diagonal moves from bishops (even though moving bishops diagonally is
    illegal under the variant)
    If filter_if_found_previous is True, only finds the first position with possible
    bishop moves
    """
    stockfish = Stockfish(depth=STOCKFISH_DEPTH, parameters=STOCKFISH_PARAMS)
    # HACK to have stockfish play our variant, just swap out the bishops for knights
    # then later pretend the knights are bishops
    stockfish.set_fen_position("rnnqknnr/pppppppp/8/8/8/8/PPPPPPPP/RNNQKNNR w KQkq - 0 1")
    previous_moves = []
    player_id = "W"

    # Get ELO of each player
    elos = [1350, 1000]
    random.shuffle(elos)
    white_elo, black_elo = elos

    bishop_diagonal_moves = []
    for _ in range(max_moves):
        if player_id == "W":
            stockfish.set_elo_rating(white_elo)
        else:
            stockfish.set_elo_rating(black_elo)

        # Find all diagonal bishop moves from this position
        found_moves = get_bishop_diagonal_moves(variant_controller, player_id)
        if len(found_moves) > 0:
            bishop_diagonal_moves.append(
                {
                    "game_idx": game_idx,
                    "previous_moves": copy.deepcopy(previous_moves),
                    "next_filtered_moves": found_moves,
                }
            )
            if filter_if_found_previous:
                break

        move = get_stockfish_move(stockfish, STOCKFIAH_MOVES_CONSIDERED)
        stockfish.make_moves_from_current_position([move])

        # Parse into notation that is compatible with my framework
        move = parse_stockfish_move(variant_controller, move)
        variant_controller.update_board(move)

        player_id = "B" if player_id == "W" else "W"
        previous_moves.append(move)

        # If checkmate or stalemate, end
        if len(variant_controller.get_player_legal_moves(player_id)) == 0:
            break

    return bishop_diagonal_moves


def create_bishop_diagonal_dataset(
    target_num_examples: int,
    max_moves: int,
    filter_if_found_previous: bool,
    filter_for_unique_previous_moves: bool,
    continuously_save: bool,
    out_path: Optional[str],
) -> Sequence[dict]:
    """
    Simulates stockfish games and finds possible diagonal moves that could be
    made by bishops.
    If filter_if_found_previous is True, finds the first move that satisfies this
    criteria in each game
    If filter_for_unique_previous_moves is True, filters to ensure each
    example has a unique set of previous moves
    If continuously_save is True, saves dataset everytime it is updated
    """
    dataset = []
    unique_previous_moves = set()

    t_bar = tqdm(total=target_num_examples)
    game_idx = 0
    while True:
        _, variant_controller, _ = initialise_boards()
        filtered_moves = find_specific_moves_in_game(
            game_idx,
            variant_controller,
            filter_if_found_previous,
            max_moves,
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
    dataset_name = "diagonal_moves_dataset.jsonl"
    out_path = os.path.join(args.out_dir, dataset_name)
    dataset = create_bishop_diagonal_dataset(
        target_num_examples=args.n_moves,
        max_moves=args.max_moves,
        filter_if_found_previous=args.filter_if_found_previous,
        filter_for_unique_previous_moves=args.filter_for_unique_previous_moves,
        continuously_save=args.continuously_save,
        out_path=out_path,
    )
    dump_sequence_to_jsonl(dataset, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--n_moves", type=int, default=5000)
    parser.add_argument("--max_moves", type=int, default=50)
    parser.add_argument(
        "--out_dir", type=str, default="./evals/registry/data/cant_do_that_anymore/"
    )
    parser.add_argument(
        "--filter_if_found_previous",
        action="store_true",
        help="Whether to filter out moves that have had previous moves that satisfy the filtering condition",
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
