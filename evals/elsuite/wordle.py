import numpy as np
from math import log2
import json
import copy
from random import Random
import evals
import evals.metrics
from typing import Any, List, Tuple, Dict

class Wordle(evals.Eval):
    def __init__(self, samples_jsonl, words_jsonl, **kwargs):
        super().__init__(**kwargs)
        path_to_word_data = words_jsonl
        word_data = self.load_jsonl_to_list(path_to_word_data)
        self.word_list = [entry["word"] for entry in word_data]
        self.entropy_scores = {entry["word"]: entry["score"] for entry in word_data}
        self.position_scores = {entry["word"]: entry["score_pos"] for entry in word_data}
        self.samples_jsonl = samples_jsonl

    def eval_sample(self, sample: Any, *_):
        def is_valid_guess(guess: str, previous_guesses: List[str], previous_feedback: List[str]) -> bool:
            for prev_guess, prev_feedback in zip(previous_guesses, previous_feedback):
                for letter, feedback in zip(prev_guess, prev_feedback):
                    if feedback == 'X' and letter in guess:
                        return False
            return True
        
        guesses = sample["guesses"]
        feedback = sample["results"]
        sol = sample["solution"]

        prompt = self.create_prompt(guesses, feedback)

        result = self.completion_fn(prompt)

        guess = result.get_completions[0]

        information_gain = self.calculate_information_gain(guess, guesses, feedback, sol)
        entropy_score = self.entropy_scores.get(guess, 0.0)
        position_score = self.position_scores.get(guess, 0.0)
        valid_guess = is_valid_guess(guess, guesses, feedback)

        evals.record.record_event('guess', information_gain=information_gain, entropy_score=entropy_score, position_score=position_score, valid_guess=valid_guess)
        

    def run(self, recorder):
        samples = self.get_samples()
        self.eval_all_samples(samples, recorder)
        events = recorder.get_events('guess')
        info_gained = [event.data['information_gain'] for event in events]
        entropy = [event.data['entropy_score'] for event in events]
        position_score = [event.data['position_score'] for event in events]

        return {
            'information_gain': np.mean(info_gained),
            'entropy_score': np.mean(entropy),
            'position_score': np.mean(position_score)
        }


    def load_jsonl_to_list(file_path):
        data_list = []
        with open(file_path, 'r') as file:
            for line in file:
                obj = json.loads(line)
                data_list.append(obj)
        return data_list


    def calculate_information_gain(
        self,
        guess: str,
        previous_guesses: List[str],
        previous_feedback: List[str],
        solution: str
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

        def calculate_entropy(words: List[str]) -> float:
            word_count = len(words)
            if word_count <= 1:
                return 0
            probability = 1.0 / word_count
            return -word_count * probability * log2(probability)
        
        def get_actual_feedback(guess: str, solution: str) -> str:
            feedback = ''
            for g, s in zip(guess, solution):
                if g == s:
                    feedback += 'V'
                elif g in solution:
                    feedback += 'O'
                else:
                    feedback += 'X'
            return feedback

        word_list = copy.copy(self.word_list)
        # Filter the word list based on previous guesses and feedback
        for prev_guess, prev_feedback in zip(previous_guesses, previous_feedback):
            word_list = [word for word in word_list if is_compatible(word, prev_guess, prev_feedback)]

        # Calculate initial entropy
        initial_entropy = calculate_entropy(self.word_list)

        # Calculate the actual feedback for the guess based on the solution
        actual_feedback = get_actual_feedback(guess, solution)

        # Get the set of words compatible with the actual feedback
        remaining_words = [word for word in self.word_list if is_compatible(word, guess, actual_feedback)]

        # Calculate the entropy of the remaining words
        remaining_entropy = calculate_entropy(remaining_words)

        # Information gain is the difference between initial entropy and remaining entropy
        information_gain = initial_entropy - remaining_entropy

        return information_gain
    
    def create_prompt(self, guesses, feedback):
        prompt = """You are a helpful assistant that helps solve a Wordle word puzzle.
    Your job is to guess the word that the puzzle is trying to solve.
    You are given a list of words that have already been guessed, along with the feedback received from each guess.

    The feedback is formatted as a 5-character string, where each character represents a letter in the word.
    The character can be one of the following:
    - X = letter not found
    - O = letter is found but it's not in the right position
    - V = letter is in the right position
    For example, if the result is XOXXV, it means that:
    - the first letter is not found and you should not use it again
    - the second letter is found and you should place it in a different position
    - the third letter is not found and you should not use it again
    - the fourth letter is not found and you should not use it again
    - the fifth letter is found and you should always use it again in the same position

    Here are the guesses so far this round, along with the feedback received:\n"""
        
        # Loop through the guesses and feedback, format them as strings, and add them to the prompt
        for guess, fb in zip(guesses, feedback):
            prompt += f"Guess: {guess}\nFeedback: {fb}\n"

        prompt += '\nYour goal is to provide a guess that will receive the feedback "VVVVV" (all letters are in the right position).\n'
        prompt += 'Try to gain as much information as possible from the feedback you receive.\n'
        prompt += 'Your ONLY response should be a 5-letter word that you want to guess next.\n\nGuess:'
        
        return prompt

if __name__ == "__main__":

    with open('examples.jsonl', 'r') as f:
        examples = [json.loads(line) for line in f]

    # Extract guesses, results, and word data from examples
    guesses = [example['guesses'] for example in examples]
    results = [example['results'] for example in examples]
    word_data = examples[0]['word_list']
    
