import json
import random

class HanabiHelper:
    def __init__(
        self,
        hand_sizes=[3, 4, 5],
        colors=["Yellow", "Red", "Green", "Blue", "Purple"],
        numbers=[1, 2, 3, 4, 5],
    ):
        self.hand_sizes = hand_sizes
        self.colors = colors
        self.numbers = numbers

    def generate_hand_with_clued_info(self, hand_size):
        if hand_size not in self.hand_sizes:
            return "Invalid hand size."

        deck = [(color, number) for color in self.colors for number in self.numbers]
        deck = deck * 2
        deck += [(color, 1) for color in self.colors]  # Add the extra 1s, as there are three of these per color
        deck = [card for card in deck if card[1] != 5] + [
            (color, 5) for color in self.colors
        ]  # this makes sure there's only one 5 per color

        random.shuffle(deck)
        hand = random.sample(deck, hand_size)

        hand_with_info = [
            {
                "color": color,
                "number": number,
                "position": pos + 1,
                "already_clued": False, # a card newly-added to hand has not yet been clued
            }
            for pos, (color, number) in enumerate(hand)
        ]

        return json.dumps(hand_with_info)

    def get_valid_clues(self, hand_json):
        hand = json.loads(hand_json)

        unique_colors = set()
        unique_numbers = set()

        for card in hand:
            unique_colors.add(card["color"])
            unique_numbers.add(str(card["number"]))

        valid_clues = list(unique_colors) + list(unique_numbers)

        return valid_clues

    def get_excluded_clues(self, valid_clues):
        all_possible_clues = set(self.colors + [str(n) for n in self.numbers])
        excluded_clues = all_possible_clues - set(valid_clues)
        return list(excluded_clues)

    def transform_hand_to_eval_format(self, hand_json, valid_clues):
        system_content = "In the game Hanabi, you will be part of a team trying to build fireworks by playing cards in ascending numerical order per color. A crucial part of the game is giving clues to teammates about their hands. You will receive a JSON object representing a hand of cards, which will include their colors and numbers. A valid clue is any color or number that appears in at least one card in the hand. The possible colors are Yellow, Red, Green, Blue, Purple, and the possible numbers are 1, 2, 3, 4, 5. Your task is to list all valid clues for the given hand in the JSON object you see. Begin your response with 'Valid clues:' and list them one by one. Make sure to capitalize the first letter of color words and use number form for the numbers (e.g., 1, 3). Please DO NOT list any invalid clues (i.e., colors or numbers that are not present) in any form, not even if you indicate that they are invalid when you do so."
        input_content = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"'{hand_json}'"},
        ]

        ideal_clues = [
            clue.capitalize() if clue.isalpha() else clue for clue in valid_clues
        ]
        excluded_clues = self.get_excluded_clues(valid_clues)

        return {"input": input_content, "ideal": ideal_clues, "exclude": excluded_clues}


# Example usage
if __name__ == "__main__":
    helper = HanabiHelper()
    random_hand = helper.generate_hand_with_clued_info(5)
    print(f"Random hand: {random_hand}")

    valid_clues = helper.get_valid_clues(random_hand)
    print(f"Valid clues: {valid_clues}")

    excluded_clues = helper.get_excluded_clues(valid_clues)
    print(f"Excluded clues: {excluded_clues}")

    transformed_hand = helper.transform_hand_to_eval_format(random_hand, valid_clues)
    print(f"Transformed hand: {json.dumps(transformed_hand, indent=4)}")
