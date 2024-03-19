import re
from abc import abstractmethod
from typing import Sequence

from evals.elsuite.cant_do_that_anymore.chess.utils import Move, parse_piece

letters = ["a", "b", "c", "d", "e", "f", "g", "h"]
letter_to_num = {i: idx for (idx, i) in enumerate(letters)}
num_to_letter = {idx: i for (idx, i) in enumerate(letters)}


def row_idx_swap(n: int) -> int:
    return 8 - n


def coord_str_to_pos(s: str) -> Sequence[int]:
    return [
        8 - int(s[1]),
        letter_to_num[s[0]],
    ]


def coord_pos_to_str(s: str) -> str:
    a = num_to_letter[s[1]]
    b = 8 - s[0]
    return f"{a}{b}".upper()


class NotationParser:
    def __init__(self, piece_str_to_id, piece_id_to_str) -> None:
        self.piece_str_to_id = piece_str_to_id
        self.piece_id_to_str = piece_id_to_str

    @abstractmethod
    def _str_to_move(self, s: str, board_state: Sequence[Sequence[int]], player_id: str) -> Move:
        raise NotImplementedError()

    @abstractmethod
    def _move_to_str(self, move: Move, board_state: Sequence[Sequence[int]], player_id: str) -> str:
        raise NotImplementedError()


class AlgebraicNotationParser(NotationParser):
    """
    Converts between coordinates of the board and algebraic notation [0]. The exact implementation
    is consistent with the python-chess library

    The regex pattern matches the following groups:
    (1) Letter indicating piece to be moved (unused)
    (2) Row of piece to be moved
    (3) Column of piece to be moved
    (4) Row+column of where piece is being moved
    (5) Letter indicating what piece the current piece is being promoted to
    (6) Special characters indicating status of game (unused)

    [0] https://en.wikipedia.org/wiki/Algebraic_notation_(chess)
    [1] https://github.com/niklasf/python-chess
    """

    pattern = re.compile(r"([a-h])([1-8])([a-h][1-8])(=?[nbrqkNBRQK])?")

    def _str_to_move(self, s: str, board_state: Sequence[Sequence[int]]) -> Move:
        match = self.pattern.match(s)
        if match is None:
            raise ValueError(
                f"Incorrect notation for move! Full start and end position must be given. Using algebraic notation, got: {s}"
            )

        # Parse start coord
        start_row = row_idx_swap(int(match.group(2))) if match.group(2) is not None else None
        start_col = letter_to_num[match.group(1)] if match.group(1) is not None else None
        start_coord = [start_row, start_col]

        # Parse to coord
        to_row = row_idx_swap(int(match.group(3)[1]))
        to_col = letter_to_num[match.group(3)[0]]
        to_coord = [to_row, to_col]

        # Promotions
        promotion = match.group(4)
        if promotion is not None:
            promotion = self.piece_str_to_id[promotion]

        # Castling
        castling = False
        if start_row is not None and start_col is not None:
            _, piece_id = parse_piece(board_state, start_row, start_col)
            if piece_id == 5 and abs(start_col - to_col) == 2:
                castling = True

        return Move(start_coord, to_coord, promotion, castling)

    def _move_to_str(self, move: Move, board_state: Sequence[Sequence[int]]) -> str:
        out_str = ""
        start_coord, target_coord = move.start_coord, move.target_coord

        start = f"{num_to_letter[start_coord[1]]}{row_idx_swap(start_coord[0])}".lower()
        out_str += start

        target = f"{num_to_letter[target_coord[1]]}{row_idx_swap(target_coord[0])}".lower()
        out_str += target

        if move.promotion is not None:
            out_str += self.piece_id_to_str[move.promotion]

        return out_str
