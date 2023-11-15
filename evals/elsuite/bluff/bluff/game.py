from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from evals.elsuite.bluff.bluff.players import Player

from evals.elsuite.bluff.bluff.cards import PlayerCards
from evals.elsuite.bluff.bluff.round import BluffMove, BluffRound
from evals.elsuite.bluff.bluff.task_description import task_description


class Game:
    def __init__(
        self, num_rounds: int, starting_player: int, rng: Optional[np.random.Generator] = None
    ):
        self.num_rounds = num_rounds
        self.starting_player = starting_player
        self.rounds: list[BluffRound] = []
        self.players: list[Player] = []
        self.rng = rng or np.random.default_rng()

    @property
    def task_description(self):
        return task_description

    def play(self):
        assert len(self.players) == 2, "Must have 2 players to play"
        for round_ix in range(self.num_rounds):
            player_1_cards, player_2_cards = self._deal_cards()

            round = BluffRound(player_1_cards, player_2_cards)
            self.rounds.append(round)

            player_ix = (round_ix + self.starting_player) % 2
            while not round.finished:
                player = self.players[player_ix]
                player.make_move()
                player_ix = 1 - player_ix

    def make_move(self, player: "Player", move: BluffMove) -> None:
        player_ix = self.players.index(player)
        self.rounds[-1].make_move(player_ix, move)

    def add_player(self, player: "Player"):
        assert player not in self.players, "Can't add the same player again"
        self.players.append(player)
        return len(self.players) - 1

    def player_cards(self, player: "Player") -> str:
        player_ix = self.players.index(player)
        return self.rounds[-1].cards[player_ix]

    def _deal_cards(self):
        cards = []
        for suit in "shdc":
            for card in "89TJQKA":
                cards.append(card + suit)

        self.rng.shuffle(cards)
        cards_1, cards_2 = cards[:5], cards[5:10]

        return PlayerCards(cards_1), PlayerCards(cards_2)
