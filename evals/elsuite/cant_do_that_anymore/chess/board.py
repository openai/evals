import copy
from typing import Callable, Dict, Sequence

from evals.elsuite.cant_do_that_anymore.chess.notation import NotationParser
from evals.elsuite.cant_do_that_anymore.chess.pieces import Piece
from evals.elsuite.cant_do_that_anymore.chess.utils import (
    Move,
    get_other_player_id,
    get_path_between_coords,
    parse_piece,
)


class Board:
    """
    Represents one board position. Is instantiated several times
    by the BoardController to simulate future boards after playing
    some moves.
    """

    def __init__(
        self,
        board_state: Sequence[Sequence[str]],
        piece_id_to_instance: Dict[int, Piece],
        piece_str_to_id: Dict[str, int],
        piece_id_to_str: Dict[int, str],
    ):
        self.board_state = board_state
        self.piece_id_to_instance = piece_id_to_instance
        self.piece_str_to_id = piece_str_to_id
        self.piece_id_to_str = piece_id_to_str

    def __str__(self) -> str:
        str_board = [["" for _ in range(8)] for _ in range(8)]

        for row_idx in range(len(self.board_state)):
            row = self.board_state[row_idx]
            for col_idx in range(len(row)):
                piece_color, piece_id = parse_piece(self.board_state, row_idx, col_idx)

                if piece_color != "E":
                    white_piece = piece_color == "W"
                    s = (
                        self.piece_id_to_instance[piece_id].white_render
                        if white_piece
                        else self.piece_id_to_instance[piece_id].black_render
                    )
                else:
                    s = "\u25A1"
                str_board[row_idx][col_idx] = s

        # Add letters on bottom
        str_board += [["-"] * 8]
        str_board += [["a", "b", "c", "d", "e", "f", "g", "h"]]

        # Add numbers on side
        str_board = [["|"] + row for row in str_board]
        numbers = list(range(8, 0, -1)) + [" ", " "]
        str_board = [[str(numbers[idx])] + row for (idx, row) in enumerate(str_board)]

        # Render as string
        str_board = "\n".join([" ".join(row) for row in str_board])
        return str_board

    def _update_board(self, move: Move):
        """
        Updates board_state according to given move. This move must have previously been checked
        to be legal. Edge cases for moves that:
        1) Take pieces at other positions where this piece isn't moving (en passant)
        2) Move two pieces (castling)
        3) Change the id of the piece (promotion)
        """
        start_coord, target_coord = move.start_coord, move.target_coord
        piece_color, piece_id = parse_piece(self.board_state, start_coord[0], start_coord[1])
        target_piece_color, target_piece_id = parse_piece(
            self.board_state, target_coord[0], target_coord[1]
        )

        # En passant
        if piece_id == 0 and target_piece_color == "E":
            dy = target_coord[1] - start_coord[1]
            target_en_passant_piece = [start_coord[0], start_coord[1] + dy]
            self.board_state[target_en_passant_piece[0]][target_en_passant_piece[1]] = "E"

        # Castling
        if move.castling:
            path = get_path_between_coords(start_coord, target_coord)
            rook_tile = path[0]
            self.board_state[rook_tile[0]][rook_tile[1]] = f"{piece_color}3"

            kingside = target_coord[1] <= 4
            old_rook_tile = [start_coord[0], 0] if kingside else [start_coord[0], 7]
            self.board_state[old_rook_tile[0]][old_rook_tile[1]] = "E"

        # Move piece
        self.board_state[start_coord[0]][start_coord[1]] = "E"
        self.board_state[target_coord[0]][target_coord[1]] = f"{piece_color}{piece_id}"

        # Promotion
        if move.promotion is not None:
            self.board_state[target_coord[0]][target_coord[1]] = f"{piece_color}{move.promotion}"

    def _get_player_moves(self, player_id: str, previous_moves: Sequence[Move]) -> Sequence[Move]:
        """
        Returns all possible moves by pieces for a player. Doesn't filter out moves that
        result in the king being placed under check
        """
        moves = []
        for row_idx in range(len(self.board_state)):
            row = self.board_state[row_idx]
            for col_idx in range(len(row)):
                piece_color, piece_id = parse_piece(self.board_state, row_idx, col_idx)
                if piece_color != player_id:
                    continue

                piece = self.piece_id_to_instance[piece_id]
                possible_piece_moves = piece.get_piece_moves(
                    self.board_state, player_id, [row_idx, col_idx], previous_moves
                )
                moves += possible_piece_moves

        return moves

    def _is_king_in_check(self, player_id: str) -> bool:
        other_player_id = get_other_player_id(player_id)

        other_player_moves = self._get_player_moves(other_player_id, [])
        king_capturing_moves = self._filter_for_king_capturing_moves(other_player_moves, player_id)
        return len(king_capturing_moves) != 0

    def _filter_for_king_capturing_moves(
        self, moves: Sequence[Move], king_color: str
    ) -> Sequence[Move]:
        king_capturing_moves = []
        for move in moves:
            piece_color, piece_id = parse_piece(
                self.board_state, move.target_coord[0], move.target_coord[1]
            )
            if piece_color == king_color and piece_id == 5:
                king_capturing_moves.append(move)

        return king_capturing_moves


class BoardController:
    """
    Manages a single game of chess. Contains logic to find all legal
    moves for a particular player and update the internal board according
    to a given move. Maintains one Board obj to represent the true state of play
    """

    def __init__(
        self,
        board_init: Callable[..., Sequence[Sequence[str]]],
        piece_id_to_instance: Dict[int, Piece],
        piece_str_to_id: Dict[str, int],
        piece_id_to_str: Dict[int, str],
        notation_parser: NotationParser,
    ):
        self.board = Board(board_init(), piece_id_to_instance, piece_str_to_id, piece_id_to_str)
        self.notation_parser = notation_parser

        self.previous_moves = []

    def __str__(self) -> str:
        return self.board.__str__()

    def update_board(self, move: str):
        """
        Parses move, updates the internal board state, then stores the move
        since knowing previous moves is necessary for En Passant and castling
        """
        move = self.notation_parser._str_to_move(move, self.board.board_state)
        self.board._update_board(move)
        self.previous_moves.append(move)

    def get_player_legal_moves(self, player_id: str) -> Sequence[str]:
        """
        Gets all legal moves for a player with the given player_id, returned in
        the notation this object was initialised with
        """
        legal_moves = self.board._get_player_moves(player_id, self.previous_moves)
        legal_moves = self._filter_to_prevent_pinning(legal_moves, player_id)

        legal_moves = [
            self.notation_parser._move_to_str(i, self.board.board_state) for i in legal_moves
        ]
        return legal_moves

    def _filter_to_prevent_pinning(self, moves: Sequence[Move], player_id: str) -> Sequence[Move]:
        """
        Filter out moves that would result in the king being pinned, or the king moving over a pinned
        position when castling
        """

        def _is_valid_castling(move: Move) -> bool:
            if self.board._is_king_in_check(player_id):
                return False

            # Check that the king won't move over an attacked position
            dy = (move.target_coord[1] - move.start_coord[1]) / abs(
                move.target_coord[1] - move.start_coord[1]
            )
            king_path = get_path_between_coords(
                move.start_coord, [move.target_coord[0], move.target_coord[1] + dy]
            )

            not_pinned_along_path = []
            for coord in king_path:
                simulated_board = copy.deepcopy(self.board)
                simulated_board._update_board(
                    Move(move.start_coord, coord, promotion=None, castling=False)
                )
                pinned = simulated_board._is_king_in_check(player_id)
                not_pinned_along_path.append(not pinned)

            if all(not_pinned_along_path):
                return True

            return False

        filtered_moves = []
        for move in moves:
            if move.castling and _is_valid_castling(move):
                filtered_moves.append(move)
            elif not move.castling:
                simulated_board = copy.deepcopy(self.board)
                simulated_board._update_board(move)
                if not simulated_board._is_king_in_check(player_id):
                    filtered_moves.append(move)

        return filtered_moves

    def _is_checkmate(self, player_id: str) -> bool:
        legal_moves = self.get_player_legal_moves(player_id)
        if len(legal_moves) == 0 and self.board._is_king_in_check(player_id):
            return True
        return False

    def _is_stalemate(self, player_id: str) -> bool:
        legal_moves = self.get_player_legal_moves(player_id)
        if len(legal_moves) == 0 and not self.board._is_king_in_check(player_id):
            return True
        return False
