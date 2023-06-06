import itertools
import random
import json
from treys import Card, Evaluator, Deck

def randomize_num_players_and_community_cards() -> tuple:
    """Randomly generate the number of players and community cards for a game.

    Returns:
        tuple: A tuple containing the number of players and community cards.
    """
    num_players = random.randint(2, 9)
    num_community_cards = random.choice([3, 4, 5])
    return num_players, num_community_cards

def generate_hands(num_players: int, num_community_cards: int) -> tuple:
    """Generate hole cards for each player and community cards.

    Args:
        num_players (int): Number of players in the game.
        num_community_cards (int): Number of community cards to generate.

    Returns:
        tuple: A tuple containing a list of hole cards for each player and a list of community cards.
    """
    deck = Deck()
    hole_cards = []
    community_cards = []

    if num_community_cards > 0:
        community_cards = deck.draw(num_community_cards)

    for i in range(num_players):
        hole_cards.append(deck.draw(2))

    return hole_cards, community_cards

def calculate_probabilities(hole_cards_list: list, community_cards: list) -> list:
    """Calculate the winning and tie probabilities for each player.

    Args:
        hole_cards_list (list): A list of hole cards for each player.
        community_cards (list): A list of community cards.

    Returns:
        list: A list of tuples with winning and tie probabilities for each player.
    """
    deck = Deck()
    evaluator = Evaluator()

    # Remove cards already on the board and in the player's hand
    for hole_cards in hole_cards_list:
        for card in hole_cards + community_cards:
            if card in deck.cards:
                deck.cards.remove(card)

    # Generate all possible combinations of the remaining community cards
    remaining_community_cards = list(
        itertools.combinations(deck.cards, 5 - len(community_cards))
    )

    num_combinations = len(remaining_community_cards)
    num_wins = [0] * len(hole_cards_list)
    num_ties = [0] * len(hole_cards_list)

    for remaining_cards in remaining_community_cards:
        full_community_cards = community_cards + list(remaining_cards)
        scores = [
            evaluator.evaluate(hole_cards + full_community_cards, [])
            for hole_cards in hole_cards_list
        ]
        best_score = min(scores)
        winners = [i for i, score in enumerate(scores) if score == best_score]

        if len(winners) == 1:
            num_wins[winners[0]] += 1
        else:
            for winner in winners:
                num_ties[winner] += 1

    win_probabilities = [win / num_combinations for win in num_wins]
    tie_probabilities = [tie / num_combinations for tie in num_ties]

    return list(zip(win_probabilities, tie_probabilities))

def generate_example(num_players: int = None, num_community_cards: int = None) -> dict:
    """Generate an example of a Texas Hold'em hand with winning probabilities.

    Args:
        num_players (int, optional): Number of players in the game. Defaults to a random value.
        num_community_cards (int, optional): Number of community cards. Defaults to a random value.

    Returns:
        dict: A dictionary containing the example input and ideal output.
    """
    (
        random_num_players,
        random_num_community_cards,
    ) = randomize_num_players_and_community_cards()
    if not num_players:
        num_players = random_num_players
    if not num_community_cards:
        num_community_cards = random_num_community_cards
    hole_cards, community_cards = generate_hands(num_players, num_community_cards)
    probabilities = calculate_probabilities(hole_cards, community_cards)

    highest_probability = max(probabilities, key=lambda x: x[0])
    winning_player = [
        i for i, prob in enumerate(probabilities) if prob == highest_probability
    ][0]

    return format_example(hole_cards, community_cards, winning_player)

def format_example(hole_cards: list, community_cards: list, winning_player: int) -> dict:
    """Format the example as a dictionary with input and ideal output.

    Args:
        hole_cards (list): A list of hole cards for each player.
        community_cards (list): A list of community cards.
        winning_player (int): The index of the player with the highest winning probability.

    Returns:
        dict: A dictionary containing the example input and ideal output.
    """
    hole_cards_str = ", ".join(
        [
            f"Player {i + 1} Hole: ({Card.int_to_str(cards[0])}, {Card.int_to_str(cards[1])})"
            for i, cards in enumerate(hole_cards)
        ]
    )
    community_str = "Community: " + ", ".join(
        [Card.int_to_str(card) for card in community_cards]
    )
    example = {
        "input": [
            {
                "role": "system",
                "content": "TASK: You will prompted with a texas hold'em hand. Which player has the highest probability of winning the hand? Answer with exactly one number 1-9 and no additional information or context.",
            },
            {"role": "user", "content": f"{hole_cards_str}, {community_str}"},
        ],
        "ideal": str(winning_player + 1),
    }
    return example

def save_examples_to_json_file(
    output_file: str, num_examples: int = 100, num_players: int = None, num_community_cards: int = None
):
    """Save examples to a JSON Lines file.

    Args:
        output_file (str): The file path for saving examples.
        num_examples (int, optional): Number of examples to generate. Defaults to 100.
        num_players (int, optional): Number of players in the game. Defaults to None.
        num_community_cards (int, optional): Number of community cards. Defaults to None.
    """
    examples = [
        generate_example(num_players, num_community_cards) for _ in range(num_examples)
    ]

    with open(output_file, "w") as f:
        for example in examples:
            f.write(json.dumps(example))
            f.write("\n")

if __name__ == "__main__":
    # This example generates 100 Texas Hold'em hands with 2 players and 5 community cards
    # and saves the input and ideal output as examples in a JSON Lines file called "samples.jsonl".
    save_examples_to_json_file(
        "samples.jsonl",
        num_examples=100,
        num_players=2,
        num_community_cards=5,
    )

