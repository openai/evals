"""
This file defines utilities for adding multiple choice questions to prompts.
"""
import random
from typing import Optional


def make_abc(answers, *, correct_idx=0, shuffle=True, rng: Optional[random.Random] = None):
    """
    ARGS
    ====
    `answers`: A sequence of strings, each of which is an answer choice.
    `correct_idx`: The integer index of the correct answer.
    `shuffle`: If True, shuffle the answer choices in the returned string.
    `rng`: If `shuffle` is True, this is the random number generator to use.

    RETURNS
    =======
    A tuple of (options, correct_answer) where `options` is a string of
    newline-separated answer choices (e.g., "A) blah") and `correct_answer` is
    the correct answer as a single character (e.g., "A").
    """

    p = list(range(len(answers)))
    if shuffle:
        if rng is None:
            raise ValueError("shuffle=True requires rng")
        rng.shuffle(p)
    options = ""
    for i, j in enumerate(p):
        if i > 0:
            options += "\n"
        options += chr(ord("A") + i) + ") " + answers[j]
    return options, chr(ord("A") + p.index(correct_idx))
