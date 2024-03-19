from dataclasses import dataclass
from typing import Sequence


@dataclass
class Move:
    start_coord: Sequence[int]
    target_coord: Sequence[int]
    promotion: int  # Either None for no promotion, or int for piece id of promotion
    castling: bool


def get_other_player_id(this_player_id: str) -> str:
    if this_player_id == "W":
        return "B"
    elif this_player_id == "B":
        return "W"
    else:
        raise ValueError(f"this_player_id var must be 'W' or 'B', but is: {this_player_id}")


def parse_piece(
    board_state: Sequence[Sequence[int]], row_idx: int, col_idx: int
) -> tuple[str, int]:
    """
    Returns the color and id of the piece at the given coords.
    """
    piece = board_state[row_idx][col_idx]
    if piece == "E":
        return "E", -1

    color = piece[0]
    id = piece[1]
    return color, int(id)


def move_crosses_pieces(board_state: Sequence[Sequence[int]], move: Move) -> bool:
    path = get_path_between_coords(move.start_coord, move.target_coord)
    for (x1, y1) in path:
        if board_state[x1][y1] != "E":
            return True

    return False


def has_piece_been_moved(
    piece_coord: Sequence[Sequence[int]], previous_moves: Sequence[Move]
) -> bool:
    for move in previous_moves:
        if move.start_coord == piece_coord:
            return True
        if move.target_coord == piece_coord:
            return True
    return False


def coord_within_board(row_idx: int, col_idx: int) -> bool:
    if row_idx < 0 or row_idx > 7:
        return False
    if col_idx < 0 or col_idx > 7:
        return False

    return True


def move_within_board(move: Move) -> bool:
    target_coord = move.target_coord
    return coord_within_board(target_coord[0], target_coord[1])


def get_path_between_coords(
    start_coord: Sequence[int], target_coord: Sequence[int]
) -> Sequence[Sequence[int]]:
    # Unpack the start and end points
    x1, y1 = start_coord
    x2, y2 = target_coord

    # Determine the steps to take in each direction
    dx = 1 if x2 > x1 else -1 if x2 < x1 else 0
    dy = 1 if y2 > y1 else -1 if y2 < y1 else 0

    path = [(x1, y1)]
    while (x1, y1) != (x2, y2):
        if x1 != x2:
            x1 += dx
        if y1 != y2:
            y1 += dy
        path.append((x1, y1))

    path = path[1:-1]
    return path


def same_color_piece_at_move_start(
    board_state: Sequence[Sequence[int]], move: Move, player_color: str
) -> bool:
    start_coord = move.start_coord
    piece_color, _ = parse_piece(board_state, start_coord[0], start_coord[1])
    return player_color == piece_color


def capturing_same_color(board_state: Sequence[Sequence[int]], move: Move) -> bool:
    start_coord, target_coord = move.start_coord, move.target_coord
    start_piece_color, _ = parse_piece(board_state, start_coord[0], start_coord[1])
    target_piece_color, _ = parse_piece(board_state, target_coord[0], target_coord[1])

    return start_piece_color == target_piece_color
