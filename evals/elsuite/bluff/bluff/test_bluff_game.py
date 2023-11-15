import pytest

from evals.elsuite.bluff.bluff.cards import PlayerCards, get_bluff_move
from evals.elsuite.bluff.bluff.round import BluffRound


#   -1: illegal move
#   0/1: winner (player cards in the code)
@pytest.mark.parametrize(
    "sequence, expected",
    (
        (("bluff",), -1),
        (("KK", "bluff"), 0),
        (("KK", "QQ"), -1),
        (("KK", "AA", "bluff"), 0),
        (("QQ", "KK", "bluff"), 1),
        (("KKKQQ", "bluff"), 0),
        (("QQQKK", "bluff"), 1),
    ),
)
def test_bluff_rules(sequence, expected):
    player_1_cards = PlayerCards("As Kh Qd Jd Td".split())
    player_2_cards = PlayerCards("Ks 9d 8d Kc Qc".split())
    round = BluffRound(player_1_cards, player_2_cards)

    player_ix = 0
    for move in sequence[:-1]:
        move = get_bluff_move(move)
        round.make_move(player_ix, move)
        player_ix = 1 - player_ix

    if expected == -1:
        with pytest.raises(ValueError):
            round.make_move(player_ix, get_bluff_move(sequence[-1]))
    else:
        round.make_move(player_ix, get_bluff_move(sequence[-1]))
        assert round.winner == expected
