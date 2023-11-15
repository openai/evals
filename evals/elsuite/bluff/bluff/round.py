from typing import Optional, Union

from evals.elsuite.bluff.bluff.cards import BluffMove, PlayerCards, get_bluff_move


class BluffRound:
    """Bluff rules enforcement & evaluation"""

    def __init__(self, player_1_cards: PlayerCards, player_2_cards: PlayerCards):
        self.cards = (player_1_cards, player_2_cards)
        self.moves: list[tuple[int, BluffMove]] = []
        self.winner: Optional[bool] = None

    @property
    def finished(self) -> bool:
        return self.winner is not None

    def summary(self) -> tuple[int, int, tuple[PlayerCards, PlayerCards]]:
        assert self.winner is not None, "summary is available only for a finished round"
        bluff_player_ix = 1 - self.moves[-1][0]
        return self.winner, bluff_player_ix, self.cards

    def make_move(self, player_ix: int, move: Union[BluffMove, str]):
        if isinstance(move, str):
            move = get_bluff_move(move)

        assert player_ix in (0, 1), "player_ix should be either 0 or 1"
        if self.winner is not None:
            raise ValueError("This round is already finished")

        if self.moves and self.moves[-1][0] == player_ix:
            raise ValueError(f"Player {player_ix} played previous move")

        if not self.moves and move == "bluff":
            raise ValueError("Can't call bluff in the first move")

        if move == "bluff":
            self._finalize()
        else:
            if self.moves and move <= self.moves[-1][1]:
                raise ValueError(f"Move {move} is below the previous move {self.moves[-1][1]}")

            self.moves.append((player_ix, move))

    def _finalize(self):
        player, last_bid = self.moves[-1]
        found = last_bid.evaluate(*self.cards)
        if found:
            self.winner = player
        else:
            self.winner = 1 - player
