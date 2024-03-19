# Default initialization
from evals.elsuite.cant_do_that_anymore.chess.pieces import Piece

# Generic type of moves
STRAIGHT_MOVES = [[0, i] for i in range(-8, 9)] + [[i, 0] for i in range(-8, 9)]
DIAGONAL_MOVES = [[i, i] for i in range(-8, 9)] + [[-i, i] for i in range(-8, 9)]

# Piece-specific moves
PAWN_MOVES_WHITE = [
    [-1, 0],
]
PAWN_MOVES_BLACK = [
    [1, 0],
]
PAWN_CAPTURING_MOVES = [
    [1, 1],
    [1, -1],
]
KNIGHT_MOVES = [
    [1, 2],
    [2, 1],
    [2, -1],
    [1, -2],
    [-1, -2],
    [-2, -1],
    [-2, 1],
    [-1, 2],
]
BISHOP_MOVES = DIAGONAL_MOVES
ROOK_MOVES = STRAIGHT_MOVES
QUEEN_MOVES = DIAGONAL_MOVES + STRAIGHT_MOVES
KING_MOVES = [
    [0, 1],
    [1, 1],
    [1, 0],
    [1, -1],
    [0, -1],
    [-1, -1],
    [-1, 0],
    [-1, 1],
]

PIECE_ID_TO_INSTANCE = {
    0: Piece(
        0,
        "\u265F",
        "\u2659",
        PAWN_MOVES_WHITE,
        PAWN_MOVES_BLACK,
        PAWN_CAPTURING_MOVES,
        can_double_step=True,
        can_en_passant=True,
        captures_like_pawn=True,
        can_promote=True,
    ),
    1: Piece(1, "\u265E", "\u2658", KNIGHT_MOVES, KNIGHT_MOVES, can_jump_over_pieces=True),
    2: Piece(
        2,
        "\u265D",
        "\u2657",
        BISHOP_MOVES,
        BISHOP_MOVES,
    ),
    3: Piece(
        3,
        "\u265C",
        "\u2656",
        ROOK_MOVES,
        ROOK_MOVES,
    ),
    4: Piece(
        4,
        "\u265B",
        "\u2655",
        QUEEN_MOVES,
        QUEEN_MOVES,
    ),
    5: Piece(5, "\u265A", "\u2654", KING_MOVES, KING_MOVES, can_castle=True),
}
# Bishops can move like knights in this variant. All other pieces play normally
VARIANT_PIECE_ID_TO_INSTANCE = {
    0: Piece(
        0,
        "\u265F",
        "\u2659",
        PAWN_MOVES_WHITE,
        PAWN_MOVES_BLACK,
        PAWN_CAPTURING_MOVES,
        can_double_step=True,
        can_en_passant=True,
        captures_like_pawn=True,
        can_promote=True,
    ),
    1: Piece(1, "\u265E", "\u2658", KNIGHT_MOVES, KNIGHT_MOVES, can_jump_over_pieces=True),
    2: Piece(
        2,
        "\u265D",
        "\u2657",
        KNIGHT_MOVES,
        KNIGHT_MOVES,
        can_jump_over_pieces=True,
    ),
    3: Piece(
        3,
        "\u265C",
        "\u2656",
        ROOK_MOVES,
        ROOK_MOVES,
    ),
    4: Piece(
        4,
        "\u265B",
        "\u2655",
        QUEEN_MOVES,
        QUEEN_MOVES,
    ),
    5: Piece(5, "\u265A", "\u2654", KING_MOVES, KING_MOVES, can_castle=True),
}
PIECE_STR_TO_ID = {"p": 0, "n": 1, "b": 2, "r": 3, "q": 4, "k": 5}
PIECE_ID_TO_STR = {0: "p", 1: "n", 2: "b", 3: "r", 4: "q", 5: "k"}
