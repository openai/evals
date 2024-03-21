import copy
from typing import Sequence

from evals.elsuite.cant_do_that_anymore.chess.utils import (
    Move,
    coord_within_board,
    get_other_player_id,
    get_path_between_coords,
    has_piece_been_moved,
    move_crosses_pieces,
    parse_piece,
)


class Piece:
    def __init__(
        self,
        piece_id: int,
        white_render: str,
        black_render: str,
        possible_moves_white: Sequence[Sequence[int]],
        possible_moves_black: Sequence[Sequence[int]],
        possible_capturing_moves: Sequence[Sequence[int]] = None,
        can_double_step: bool = False,
        can_en_passant: bool = False,
        captures_like_pawn: bool = False,
        can_promote: bool = False,
        can_jump_over_pieces: bool = False,
        can_castle: bool = False,
    ):
        self.piece_id = piece_id
        self.white_render = white_render
        self.black_render = black_render
        self.possible_moves_white = possible_moves_white
        self.possible_moves_black = possible_moves_black
        self.possible_capturing_moves = possible_capturing_moves

        self.can_double_step = can_double_step
        self.can_en_passant = can_en_passant
        self.captures_like_pawn = captures_like_pawn
        self.can_promote = can_promote
        self.can_jump_over_pieces = can_jump_over_pieces
        self.can_castle = can_castle

    def get_piece_moves(
        self,
        board_state: Sequence[Sequence[int]],
        player_id: str,
        start_coord: Sequence[int],
        previous_moves: Sequence[Move],
    ) -> Sequence[Move]:
        """
        Returns a sequence representing all moves this piece can make given the current environment
        and rules this piece follows
        """
        if player_id == "W":
            possible_transformations = copy.deepcopy(self.possible_moves_white)
            forward_direction = -1
        else:
            possible_transformations = copy.deepcopy(self.possible_moves_black)
            forward_direction = 1

        # Get all relative transformations piece can make
        if self.can_double_step:
            possible_transformations += self._get_pawn_double_step_transformations(
                player_id, start_coord
            )
        if self.captures_like_pawn:
            possible_transformations = self._remove_illegal_pawn_capture_transformations(
                board_state, player_id, start_coord, possible_transformations, forward_direction
            )
        if self.can_en_passant:
            possible_transformations += self._get_en_passant_transformations(
                board_state, start_coord, previous_moves, forward_direction
            )

        # Find all legal moves from transformations
        piece_moves = self._get_moves_from_transformations(
            board_state, player_id, start_coord, possible_transformations
        )

        # Add rule-specific moves
        if self.can_promote:
            piece_moves = self._add_promotion_moves(piece_moves)
        if self.can_castle:
            piece_moves += self._get_castling_possible_moves(board_state, player_id, previous_moves)

        return piece_moves

    def _get_moves_from_transformations(
        self,
        board_state: Sequence[Sequence[int]],
        player_id: str,
        start_coord: Sequence[int],
        possible_transformations: Sequence[Sequence[int]],
    ) -> Sequence[Move]:
        """
        Given a piece's position within a board and the set of possible relative
        transformations the piece can make, convert each transformation into a `Move`
        object if:
        1) Transformation results in piece being on board
        2) Transformation doesn't result in piece ending up on piece of same color
        3) Transformation doesn't "jump" over other pieces, unless this piece is
        allowed to do so (e.g. knight)
        """
        piece_moves = []
        for move in possible_transformations:
            new_row_idx = start_coord[0] + move[0]
            new_col_idx = start_coord[1] + move[1]

            if not coord_within_board(new_row_idx, new_col_idx):
                continue

            target_coord = [new_row_idx, new_col_idx]
            target_piece_color, target_piece_id = parse_piece(
                board_state,
                target_coord[0],
                target_coord[1],
            )
            move = Move(start_coord, target_coord, None, False)

            if target_piece_color == player_id:
                continue
            if not self.can_jump_over_pieces and move_crosses_pieces(board_state, move):
                continue

            piece_moves.append(move)

        return piece_moves

    def _get_pawn_double_step_transformations(
        self, player_id: str, start_coord: Sequence[int]
    ) -> Sequence[Sequence[int]]:
        if player_id == "W" and start_coord[0] == 6:
            return [[-2, 0]]
        elif player_id == "B" and start_coord[0] == 1:
            return [[2, 0]]
        return []

    def _remove_illegal_pawn_capture_transformations(
        self,
        board_state: Sequence[Sequence[int]],
        player_id: str,
        start_coord: Sequence[int],
        possible_transformations: Sequence[Sequence[int]],
        forward_direction: int,
    ) -> Sequence[Sequence[int]]:
        """
        Prevents pawns from "capturing forward"
        """
        if self.piece_id != 0:
            return possible_transformations

        new_possible_transformations = []
        capturing_moves = self.possible_capturing_moves
        capturing_moves = [[move[0] * forward_direction, move[1]] for move in capturing_moves]
        for move in possible_transformations + capturing_moves:
            new_row_idx = start_coord[0] + move[0]
            new_col_idx = start_coord[1] + move[1]

            if not coord_within_board(new_row_idx, new_col_idx):
                continue

            target_piece_color, target_piece_id = parse_piece(board_state, new_row_idx, new_col_idx)

            if target_piece_color == "E" and move not in capturing_moves:
                new_possible_transformations.append(move)
            elif target_piece_color == get_other_player_id(player_id) and move in capturing_moves:
                new_possible_transformations.append(move)

        return new_possible_transformations

    def _get_en_passant_transformations(
        self,
        board_state: Sequence[Sequence[int]],
        start_coord: Sequence[int],
        previous_moves: Sequence[Move],
        forward_direction: int,
    ) -> Sequence[Sequence[int]]:
        last_move = previous_moves[-1] if len(previous_moves) > 0 else None
        if last_move is not None and self.piece_id == 0:
            _, last_piece_id = parse_piece(
                board_state, last_move.target_coord[0], last_move.target_coord[1]
            )

            # If last move was pawn moving two tiles
            if (
                last_piece_id == 0
                and abs(last_move.start_coord[0] - last_move.target_coord[0]) == 2
            ):

                # If on same row and one column apart
                dx = start_coord[1] - last_move.target_coord[1]
                dy = start_coord[0] - last_move.target_coord[0]
                if dy == 0 and abs(dx) == 1:
                    return [[forward_direction, -dx]]
        return []

    def _add_promotion_moves(self, piece_moves: Sequence[Move]) -> Sequence[Move]:
        new_piece_moves = []
        for move in piece_moves:
            target_coord = move.target_coord
            if target_coord[0] == 0 or target_coord[0] == 7:
                for promotion_piece_id in [1, 2, 3, 4]:
                    move_promotion = copy.deepcopy(move)
                    move_promotion.promotion = promotion_piece_id
                    new_piece_moves.append(move_promotion)
            else:
                new_piece_moves.append(move)

        return new_piece_moves

    def _get_castling_possible_moves(
        self, board_state: Sequence[Sequence[int]], player_id: str, previous_moves: Sequence[Move]
    ) -> Sequence[Move]:
        castling_moves = []
        if self.piece_id != 5:
            return castling_moves

        def _can_pieces_castle(
            king_init_coord: Sequence[int], rook_init_coord: Sequence[int], init_rook_id: int
        ) -> Sequence[Move]:
            if init_rook_id != 3:
                return []

            if has_piece_been_moved(king_init_coord, previous_moves) or has_piece_been_moved(
                rook_init_coord, previous_moves
            ):
                return []

            king_to_rook_move = Move(king_init_coord, rook_init_coord, None, False)
            if move_crosses_pieces(board_state, king_to_rook_move):
                return []

            king_to_rook_path = get_path_between_coords(king_init_coord, rook_init_coord)
            move = Move(king_init_coord, king_to_rook_path[1], None, True)
            return [move]

        # ASSUME board init
        king_init_coord = [7, 4] if player_id == "W" else [0, 4]
        _, init_king_id = parse_piece(board_state, king_init_coord[0], king_init_coord[1])
        if init_king_id != 5:
            return castling_moves

        # Queenside
        queenside_rook_init_coord = [7, 7] if player_id == "W" else [0, 7]
        _, init_rook_id = parse_piece(
            board_state, queenside_rook_init_coord[0], queenside_rook_init_coord[1]
        )
        castling_moves += _can_pieces_castle(
            king_init_coord, queenside_rook_init_coord, init_rook_id
        )

        # Kingside
        kingside_rook_init_coord = [7, 0] if player_id == "W" else [0, 0]
        _, init_rook_id = parse_piece(
            board_state, kingside_rook_init_coord[0], kingside_rook_init_coord[1]
        )
        castling_moves += _can_pieces_castle(
            king_init_coord, kingside_rook_init_coord, init_rook_id
        )

        return castling_moves
