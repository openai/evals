from collections import defaultdict

import numpy as np

from evals.elsuite.bluff.bluff.cards import (
    CARDS,
    PlayerCards,
    get_all_winning_hands,
    get_bluff_move,
)
from evals.elsuite.bluff.bluff.game import Game


class Player:
    def __init__(self, game: Game):
        self.game = game
        self.game.add_player(self)
        self.rng = np.random.default_rng()  # Note: this might be overwritten

    @property
    def cards(self) -> PlayerCards:
        return self.game.player_cards(self)

    def make_move(self) -> None:
        raise NotImplementedError


class SimplestBot(Player):
    def make_move(self):
        if self.game.rounds[-1].moves:
            move = "bluff"
        else:
            move = "99"
        self.game.make_move(self, move)


class HonestBotLowest(Player):
    def make_move(self):
        """Bid the lowest poker hand you have in cards, or call bluff if it is too low"""
        winning_hands = get_all_winning_hands(self.cards)
        current_round = self.game.rounds[-1]
        for hand in winning_hands:
            if not current_round.moves or hand > current_round.moves[-1][1]:
                move = hand
                break
        else:
            move = "bluff"
        self.game.make_move(self, move)


class HonestBotHighest(Player):
    def make_move(self):
        """Bid the highest poker hand you have in cards, or call bluff if it is too low"""
        move = get_all_winning_hands(self.cards)[-1]
        current_round = self.game.rounds[-1]
        if current_round.moves and move <= current_round.moves[-1][1]:
            move = "bluff"
        self.game.make_move(self, move)


class StrongBot(Player):
    """This is a bot that plays pretty well (~ 0.75 winrate against GPT-4), but not very well.

    We now don't need a very strong bot, this is the best one we could implement in less than a day of work.

    Brief strategy summary:
    * In the first move there is a chance of a low random bid. The weaker the hand, the higher the chance.
    * If not first move or first move and not bluffing:
        *   If we hold a high-enough combination in hand, we bid it (if there are multiple, we bid the lowest)
        *   If not, we simulate possible opponent's hands, assuming that hands with more cards matching
            their last bid are more likely. We take the action that wins for the largest (weighted) number
            of opponent's hands.
            *   NOTE: We ignore the fact that some cards declared by the opponent might be our cards, e.g.
                      if we open with 'AA' and they respond with 'AAA' then we treat this the same way
                      as if they had opened with 'AAA'. This is how we often lose against GPT-4.

    This strategy works pretty well against GPT-4, but has a pretty strong counterstrategy (bid pure-bluff high bids,
    call bluff only when your opponent raises your bluff).
    """

    def make_move(self):
        move = self.get_move()
        self.game.make_move(self, move)

    def get_move(self):
        round = self.game.rounds[-1]

        if not round.moves or (len(round.moves) == 1 and round.moves[-1][1] < get_bluff_move("99")):
            if self.rng.random() > self._get_winning_hand_probability():
                return self._bluff_bid()
        return self._honest_bid()

    def _bluff_bid(self):
        """Returns a random low bid"""
        available_bids = [get_bluff_move(x) for x in list(CARDS) + ["88", "99", "TT", "JJ"]]
        if self.game.rounds[-1].moves:
            last_move = self.game.rounds[-1].moves[-1][1]
            available_bids = [bid for bid in available_bids if bid > last_move]
        return self.rng.choice(available_bids)

    def _honest_bid(self):
        winning_hands = get_all_winning_hands(self.cards)
        if self.game.rounds[-1].moves:
            last_move = self.game.rounds[-1].moves[-1][1]
            winning_hands = [bid for bid in winning_hands if bid > last_move]
        if winning_hands:
            return winning_hands[0]
        else:
            return self._bid_higher_or_call_bluff()

    def _bid_higher_or_call_bluff(self):
        #   More samples -> higher accuracy, but also longer - this is ~ 0.3s per bid on my laptop
        NUM_SAMPLES = 2000

        #   How honest we assume our opponent to be, 1 is "their bids say nothing about their cards".
        #   Note: logic around OPP_HONESTY_FACTOR is simple and makes some sense but don't expect
        #   anything clever here. Current value is fine-tuned against GPT-4.
        OPP_HONESTY_FACTOR = 3.5

        last_bid = self.game.rounds[-1].moves[-1][1]
        last_bid_cards = last_bid.cards()

        bid_scores = defaultdict(int)
        for _ in range(NUM_SAMPLES):
            hand = self._random_opp_hand()
            hand_cards = hand.no_suit()

            common_cards = 0
            for card in CARDS:
                common_cards += min(last_bid_cards.count(card), hand_cards.count(card))

            weight = OPP_HONESTY_FACTOR**common_cards

            winning_bids = [
                bid for bid in get_all_winning_hands(hand, self.cards) if bid > last_bid
            ]
            if not last_bid.evaluate(hand, self.cards):
                winning_bids.append("bluff")

            for bid in winning_bids:
                bid_scores[str(bid)] += weight

        best_bid = max(bid_scores.items(), key=lambda x: x[1])[0]
        return best_bid

    def _get_winning_hand_probability(self) -> float:
        """Calculate the probability that we hold a stronger combination than they do.

        E.g. for AA998 this is > 0.9, because AA99 is high, and for AKJT8 this is low, because our
        strongest combination is just a single ace.
        """
        my_best_hand = get_all_winning_hands(self.cards)[-1]

        num_hands = 100
        other_best_hands = [
            get_all_winning_hands(self._random_opp_hand())[-1] for _ in range(num_hands)
        ]
        winning_cnt = sum(my_best_hand > other_best_hand for other_best_hand in other_best_hands)
        return winning_cnt / num_hands

    def _random_opp_hand(self) -> PlayerCards:
        """Returns a random opponent hand. We take our cards into account."""
        cards = []
        for suit in "shdc":
            for card in CARDS:
                if card not in self.cards.cards[suit]:
                    cards.append(card + suit)

        self.rng.shuffle(cards)
        return PlayerCards(cards[:5])
