import random
import time
from typing import Sequence

import pytest
from tqdm import tqdm

from evals.elsuite.cant_do_that_anymore.chess.board import BoardController
from evals.elsuite.cant_do_that_anymore.chess.move_variants import (
    PIECE_ID_TO_INSTANCE,
    PIECE_ID_TO_STR,
    PIECE_STR_TO_ID,
)
from evals.elsuite.cant_do_that_anymore.chess.notation import AlgebraicNotationParser

N_GAMES = 100
MAX_MOVES = 1000
VERBOSE = False
VERBOSE_SLOWDOWN = 2


def default_board_init() -> Sequence[Sequence[str]]:
    board = [
        ["B3", "B1", "B2", "B4", "B5", "B2", "B1", "B3"],
        ["B0", "B0", "B0", "B0", "B0", "B0", "B0", "B0"],
        ["E", "E", "E", "E", "E", "E", "E", "E"],
        ["E", "E", "E", "E", "E", "E", "E", "E"],
        ["E", "E", "E", "E", "E", "E", "E", "E"],
        ["E", "E", "E", "E", "E", "E", "E", "E"],
        ["W0", "W0", "W0", "W0", "W0", "W0", "W0", "W0"],
        ["W3", "W1", "W2", "W4", "W5", "W2", "W1", "W3"],
    ]
    return board


@pytest.mark.skip  # avoid unit test that requires chess library
def simulate_games():
    """
    Simulates full chess games and asserts that at every position, the
    set of legal moves is equivalent to the legal moves reported by the
    python-chess library

    Install such library with:
    pip install chess
    """
    import chess

    for _ in tqdm(range(N_GAMES)):
        my_controller = BoardController(
            default_board_init,
            PIECE_ID_TO_INSTANCE,
            PIECE_STR_TO_ID,
            PIECE_ID_TO_STR,
            AlgebraicNotationParser(PIECE_STR_TO_ID, PIECE_ID_TO_STR),
        )
        their_controller = chess.Board()  # python-chess equivalent

        my_player_id = "W"
        for _ in range(MAX_MOVES):
            our_legal_moves = sorted(my_controller.get_player_legal_moves(my_player_id))
            their_legal_moves = sorted([str(i) for i in their_controller.legal_moves])

            if our_legal_moves != their_legal_moves:
                our_additional_moves = list(set(our_legal_moves) - set(their_legal_moves))
                their_additional_moves = list(set(their_legal_moves) - set(our_legal_moves))
                print(
                    f"""
                      Inconsistent legal moves between the boards!
                      Our legal moves: {our_legal_moves},
                      Their legal moves: {their_legal_moves},
                      Moves we had they didnt: {our_additional_moves},
                      Moves they had we didn't: {their_additional_moves},
                      Board state:\n{my_controller.board.board_state}
                      """
                )
                assert False

            if len(our_legal_moves) == 0:
                break

            # Pick random move
            move = random.choice(our_legal_moves)
            my_controller.update_board(move)
            their_controller.push_san(move)

            my_player_id = "B" if my_player_id == "W" else "W"

            if VERBOSE:
                print(my_controller)
                print(move)
                time.sleep(VERBOSE_SLOWDOWN)


if __name__ == "__main__":
    simulate_games()
