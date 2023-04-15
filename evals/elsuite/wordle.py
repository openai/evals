from itertools import product
import evals
import evals.metrics
from typing import List, Tuple

def get_word_score(
    guess: str,
    previous_guesses: List[str],
    previous_feedback: List[str],
    word_list: List[str],
) -> float:
    def is_compatible(word: str, guess: str, feedback: str) -> bool:
        feedback_calculated = ''
        for w, g in zip(word, guess):
            if w == g:
                feedback_calculated += 'V'
            elif w in guess:
                feedback_calculated += 'O'
            else:
                feedback_calculated += 'X'
        return feedback_calculated == feedback

    # Filter the word list based on previous guesses and feedback
    for prev_guess, prev_feedback in zip(previous_guesses, previous_feedback):
        word_list = [word for word in word_list if is_compatible(word, prev_guess, prev_feedback)]
    
    # Calculate the expected number of remaining words for the current guess
    possible_feedback = [''.join(p) for p in product('XOV', repeat=5)]
    expected_remaining_words = 0.0
    for feedback in possible_feedback:
        remaining_words = len([word for word in word_list if is_compatible(word, guess, feedback)])
        expected_remaining_words += remaining_words * (1.0 / len(possible_feedback))
    
    return expected_remaining_words


# Define modified get_combined_word_score function with penalty for invalid guesses
def get_combined_word_score_modified_v2(
    guess: str,
    previous_guesses: List[str],
    previous_feedback: List[str],
    word_data: List[Dict[str, float]],
    weights: Tuple[float, float, float] = (10.0, 1.0, 10.0),  # Increase weight for wordle_score
    invalid_guess_penalty: float = -1000.0,  # Large negative penalty for invalid guesses
    epsilon: float = 1e-6,  # Small constant to avoid division by zero
) -> Tuple[float, bool]:
    is_valid_guess = True
    for prev_guess, prev_feedback in zip(previous_guesses, previous_feedback):
        for letter, feedback in zip(prev_guess, prev_feedback):
            if feedback == 'X' and letter in guess:
                is_valid_guess = False
                break
    
    word_list = [entry["word"] for entry in word_data]
    entropy_scores = {entry["word"]: entry["score"] for entry in word_data}
    position_scores = {entry["word"]: entry["score_pos"] for entry in word_data}
    weight_wordle, weight_entropy, weight_position = weights
    wordle_score = get_word_score(guess, previous_guesses, previous_feedback, word_list)
    wordle_score = 1 / (wordle_score + epsilon)  # Invert wordle_score
    entropy_score = entropy_scores.get(guess, 0.0)
    position_score = position_scores.get(guess, 0.0)
    combined_score = (
        weight_wordle * wordle_score
        + weight_entropy * entropy_score
        + weight_position * position_score
    )
    # Apply penalty if guess is invalid
    if not is_valid_guess:
        combined_score += invalid_guess_penalty
    return combined_score, is_valid_guess



# # Example usage
# example_guess = "chair"
# example_previous_guesses = ["apple", "water"]
# example_previous_feedback = [(1, 1), (1, 0)]

# get_word_score(example_guess, example_previous_guesses, example_previous_feedback)


class Wordle(evals.Eval):
    def __init__(self, train_jsonl, test_jsonl, **kwargs):
        super().__init__(**kwargs)
        self.train_jsonl = train_jsonl
        self.test_jsonl = test_jsonl

def test_functions_modified_v2(guesses, results, next_guess, word_data):
    # Test get_word_score_modified function
    word_list = [entry["word"] for entry in word_data]
    word_score_modified = get_word_score(
        next_guess, 
        list(guesses), 
        list(results), 
        word_list
    )
    
    # Test get_combined_word_score_modified_v2 function
    combined_word_score, is_valid_guess = get_combined_word_score_modified_v2(
        next_guess, 
        list(guesses), 
        list(results), 
        word_data
    )
    
    # Print the results
    print("Output of get_word_score_modified function:", word_score_modified)
    print("Output of get_combined_word_score_modified_v2 function:", combined_word_score, is_valid_guess)

if __name__ == "__main__":
    guesses = ("arose", "intro")
    results = ("XOOXX", "XXVOO")
    next_guess = "tutor"
    
