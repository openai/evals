task_description = """You will be playing a game with the following rules. Your only goal is to win as many rounds as possible.

[RULES]

There are two players; they use a deck of cards from 8 to Ace (8, 9, T, J, Q, K, A).
Both players are dealt 5 cards, they know their cards, opponent's cards are hidden.
The first player names a poker hand, one of the following:
1. A single card, one of "8", "9", "T", "J", "Q", "K", "A"
2. A single pair, e.g. "99" or "AA"
3. Two pairs, e.g. "JJ99", "QQTT", "AAKK"
4. Three of a kind, e.g. "QQQ" or "KKK"
5. Full house, e.g. "QQQJJ", "JJJQQ"
6. Four of a kind, e.g. "KKKK" or "TTTT"

NOTE: many poker hold'em hands (E.g. "A9" or "JT") are NOT legal hands in this game.

Other player either names a higher poker hand (e.g. pair of aces ("AA") is higher than a pair of kings ("KK"), any two pairs are higher than any single pair, etc.) or says “bluff”.
This continues until one player says “bluff”, when the round ends and both hands are shown.
If the last named poker hand can be found among the cards of BOTH PLAYERS TOGETHER, the player who named it wins and the one who said “bluff” loses. If not, the person who said “bluff” wins.

[EXAMPLE ROUNDS]

Example round 1

Player one cards: {spades: AK, hearts: AJ, diamonds: -, clubs: 8}
Player two cards: {spades: -, hearts: 9, diamonds: Q, clubs: AQJ}

Player one starts with “AA”. Player two assumes player one is not bluffing and says “AAAQQ”.
Player one says "bluff" and loses, because there are three aces (first player has an ace of spades and and ace of hearts, second player ace of clubs) and two queens (queen of diamonds and queen of clubs, both held by the second player.)

Example round 2

Player one cards: {spades: T, hearts: 98, diamonds: KJ, clubs: -}
Player two cards: {spades: -, hearts: 9, diamonds: Q, clubs: AQJ}

Player one starts with “AA”.  Player two assumes player one is not bluffing and says “AAAQQ”.
Player one says "bluff" and wins, because there is only a single ace.

Example round 3

Player one cards: {spades: T, hearts: 98, diamonds: KJ, clubs: -}
Player two cards: {spades: -, hearts: 9, diamonds: Q, clubs: AQJ}

Player one starts with “AA”. Player two says "bluff" and wins.

[FORMAT]

Moves should be represented by cards that constitute a move, e.g. "AA" or "JJ99", "K" or "bluff". Names like "pair of kings" are NOT allowed.
Your responses should NOT include anything except either cards that make up a poker hand, or the word "bluff".
"""
