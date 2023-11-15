"""All the card-related logic is in this file (both player cards and poker hands)"""

from functools import total_ordering
from itertools import combinations
from typing import Literal, Union

BluffMove = Union["PokerHand", Literal["bluff"]]

CARDS = "89TJQKA"


class PlayerCards:
    def __init__(self, cards: list[str]):
        """In: e.g. [As, Ah, Kh, Qd, 9c]"""
        assert len(cards) == 5

        self.cards = {}
        for suit in "shdc":
            self.cards[suit] = sorted(card[0] for card in cards if card[1] == suit)

    def no_suit(self):
        return sorted(self.cards["s"] + self.cards["h"] + self.cards["d"] + self.cards["c"])

    def lm_format(self):
        return (
            "{"
            f"spades: {self._suit_repr('s')}, "
            f"hearts: {self._suit_repr('h')}, "
            f"diamonds: {self._suit_repr('d')}, "
            f"clubs: {self._suit_repr('c')}"
            "}"
        )

    def _suit_repr(self, suit):
        cards = sorted(self.cards[suit], key=lambda x: CARDS.index(x), reverse=True)
        return "".join(cards) or "-"

    def __repr__(self):
        return str(self.cards)


def get_poker_hand(txt: str) -> "PokerHand":
    """In: some text, e.g. 'AA' or 'QQJJ', out: an instance of a subclass of PokerHand"""
    hands = []
    for cls in (HighCard, OnePair, TwoPair, ThreeOfAKind, FullHouse, FourOfAKind):
        hand = cls.from_string(txt)
        if hand is not None:
            hands.append(hand)
    if len(hands) > 1:
        raise ValueError(
            f"Hand descrption {txt} fits multiple hands: {','.join([str(x) for x in hands])}"
        )
    elif len(hands) == 0:
        raise ValueError(f"Hand description {txt} doesn't describe any poker hand")
    else:
        return hands[0]


def get_bluff_move(txt: str) -> BluffMove:
    """IN: a string, out: a BluffMove (something accepted by Round.make_move())"""
    if txt.lower() == "bluff":
        return "bluff"
    return get_poker_hand(txt)


def get_all_hands():
    """Return all valid poker hands, sorted from weakest to strongest"""
    return sorted(
        HighCard.all()
        + OnePair.all()
        + TwoPair.all()
        + ThreeOfAKind.all()
        + FullHouse.all()
        + FourOfAKind.all()
    )


def get_all_winning_hands(*in_cards: PlayerCards):
    """Return all winning poker hands for a given set of cards, sorted from weakest to strongest.

    NOTE: this is equivalent to
        [hand for hand in get_all_hands() if hand.evaluate(*cards)]
    but much faster.
    """
    all_cards = []
    for cards in in_cards:
        all_cards += cards.no_suit()

    winning_hands = []
    winning_hands += [HighCard(card) for card in set(all_cards)]
    winning_hands += [OnePair(card) for card in set(all_cards) if all_cards.count(card) >= 2]
    winning_hands += [ThreeOfAKind(card) for card in set(all_cards) if all_cards.count(card) >= 3]
    winning_hands += [FourOfAKind(card) for card in set(all_cards) if all_cards.count(card) >= 4]

    pairs = [x for x in winning_hands if isinstance(x, OnePair)]
    for ix, first_pair in enumerate(pairs):
        for second_pair in pairs[ix + 1 :]:
            winning_hands.append(TwoPair(first_pair.card, second_pair.card))

    trios = [x for x in winning_hands if isinstance(x, ThreeOfAKind)]
    for trio in trios:
        for pair in pairs:
            if trio.card != pair.card:
                winning_hands.append(FullHouse(trio.card, pair.card))

    winning_hands.sort()

    return winning_hands


@total_ordering
class PokerHand:
    def __eq__(self, other):
        return isinstance(self, type(other)) and self.cards() == other.cards()

    def __lt__(self, other):
        if isinstance(other, type(self)):
            my_card_ixs = [CARDS.index(card) for card in self.cards()]
            other_card_ixs = [CARDS.index(card) for card in other.cards()]
            return my_card_ixs < other_card_ixs
        elif isinstance(other, PokerHand):
            return self.type_val < other.type_val
        raise TypeError(f"Cant compare {type(self).__name__} to {type(other).__name__}")

    def __repr__(self):
        return self.cards()

    def evaluate(self, *player_cards: PlayerCards) -> bool:
        """Check if this hand can be found in given set of cards"""
        all_cards = []
        for cards in player_cards:
            all_cards += cards.no_suit()

        all_cards.sort()
        my_cards = self.cards()
        all_combinations = list(combinations(all_cards, len(my_cards)))
        return sorted(my_cards) in [sorted(x) for x in all_combinations]


class HighCard(PokerHand):
    type_val = 0

    def __init__(self, card: str):
        self.card = card

    def cards(self) -> str:
        return self.card

    @classmethod
    def from_string(cls, txt):
        if len(txt) == 1 and txt in CARDS:
            return cls(txt)

    @classmethod
    def all(self):
        return [HighCard(x) for x in CARDS]


class OnePair(PokerHand):
    type_val = 1

    def __init__(self, card: str):
        self.card = card

    def cards(self) -> str:
        return self.card * 2

    @classmethod
    def from_string(cls, txt):
        if len(txt) == 2 and txt[0] == txt[1] and txt[0] in CARDS:
            return cls(txt[0])

    @classmethod
    def all(cls):
        return [OnePair(x) for x in CARDS]


class TwoPair(PokerHand):
    type_val = 2

    def __init__(self, card_1: str, card_2: str):
        assert card_1 != card_2, "pairs in TwoPair must be different"

        #   Higher card first
        if CARDS.index(card_1) < CARDS.index(card_2):
            card_1, card_2 = card_2, card_1

        self.card_high = card_1
        self.card_low = card_2

    def cards(self) -> str:
        return self.card_high * 2 + self.card_low * 2

    @classmethod
    def from_string(cls, txt):
        if (
            len(txt) == 4
            and txt[0] == txt[1]
            and txt[1] != txt[2]
            and txt[2] == txt[3]
            and txt[0] in CARDS
            and txt[2] in CARDS
        ):
            return cls(txt[0], txt[2])

    @classmethod
    def all(cls):
        result = []
        for card_1 in CARDS:
            for card_2 in CARDS:
                if card_1 < card_2:
                    result.append(TwoPair(card_1, card_2))
        return result


class ThreeOfAKind(PokerHand):
    type_val = 3

    def __init__(self, card: str):
        self.card = card

    def cards(self) -> str:
        return self.card * 3

    @classmethod
    def from_string(cls, txt):
        if len(txt) == 3 and txt[0] == txt[1] == txt[2] and txt[0] in CARDS:
            return cls(txt[0])

    @classmethod
    def all(cls):
        return [ThreeOfAKind(x) for x in CARDS]


class FullHouse(PokerHand):
    type_val = 4

    def __init__(self, card_triple: str, card_pair: str):
        assert card_triple != card_pair, "pair/triple in FullHouse must be different"

        self.card_triple = card_triple
        self.card_pair = card_pair

    def cards(self) -> str:
        return self.card_triple * 3 + self.card_pair * 2

    @classmethod
    def from_string(cls, in_txt):
        #   in_txt should be AAAKK, but KKAAA is also fine
        reversed_order_txt = in_txt[2:] + in_txt[:2]
        for txt in (in_txt, reversed_order_txt):
            if (
                len(txt) == 5
                and txt[0] == txt[1] == txt[2]
                and txt[2] != txt[3]
                and txt[3] == txt[4]
                and txt[0] in CARDS
                and txt[3] in CARDS
            ):
                return cls(txt[0], txt[3])

    @classmethod
    def all(cls):
        result = []
        for card_1 in CARDS:
            for card_2 in CARDS:
                if card_1 != card_2:
                    result.append(FullHouse(card_1, card_2))
        return result


class FourOfAKind(PokerHand):
    type_val = 5

    def __init__(self, card: str):
        self.card = card

    def cards(self) -> str:
        return self.card * 4

    @classmethod
    def from_string(cls, txt):
        if len(txt) == 4 and txt[0] == txt[1] == txt[2] == txt[3] and txt[0] in CARDS:
            return cls(txt[0])

    @classmethod
    def all(cls):
        return [FourOfAKind(x) for x in CARDS]
