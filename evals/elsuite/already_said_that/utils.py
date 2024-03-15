import random
import re
from typing import Any, Optional

from evals.elsuite.already_said_that.distractors import DistractorSample
from evals.task_state import Message


def build_message(
    words_not_shown: set[str],
    words_prev_shown: set[str],
    distracting_words: set[str],
    rng: random.Random,
    distractor_sample: Optional[DistractorSample] = None,
) -> tuple[Message, list[str], Optional[DistractorSample]]:
    """
    Builds the TaskState.Message for a given sample.
    Randomly chooses whether to show a word (base task) or ask a question (distractor).
    In case of base task, the words are randomly chosen either from base_words or from
    distracting_words, i.e. words that have been mentioned by the solver or as part of
    the distractor task in previous rounds.

    Args:
        words_not_shown: set of words that have not been shown to the solver
        words_prev_shown: set of words that have been shown to the solver in
        distracting_words: set of words that have been mentioned by the solver or as part of
            the distractor questions in previous rounds
        distractor_sample: the distractor sample to use in case of a distractor round
        rng: seeded random number generator, for reproducibility

    Returns:
        message: the TaskState.Message
        message_words: list of words that are part of the message
        distractor_added: The distractor sample if the message is a distractor question,
            else None
    """
    # we mainly do the base task, but sometimes we do a distractor question
    is_a_distractor_question = rng.choices([True, False], weights=[1 / 3, 2 / 3], k=1)[0]
    if not is_a_distractor_question or distractor_sample is None:
        message, message_words = build_base_task_message(
            words_not_shown, words_prev_shown, distracting_words, rng
        )
        distractor_added = None
    else:
        message, message_words = build_distractor_question_message(distractor_sample)
        distractor_added = distractor_sample
    return message, message_words, distractor_added


def build_base_task_message(
    words_not_shown: set[str],
    words_prev_shown: set[str],
    distracting_words: set[str],
    rng,
) -> tuple[Message, list[str]]:
    distract = rng.choice([True, False])
    if distract and len(distracting_words) > 0:
        word = rng.choice(list(distracting_words))
    else:
        # balance between showing new words and repeating old ones
        word_pool = rng.choice([words_not_shown, words_prev_shown])
        if len(word_pool) > 0:
            word = rng.choice(list(word_pool))
        else:
            # this is always non-empty
            word = rng.choice(list(words_not_shown))
    word = word.lower()
    message = Message(role="user", content=f"MAIN TASK - {word}")
    return message, [word]


def build_distractor_question_message(
    distractor_sample: list[str],
) -> tuple[Message, list[str]]:
    question = distractor_sample.question
    message = Message(role="user", content=question)
    question_words = find_alpha_words(question)
    return message, question_words


def find_alpha_words(s: str) -> list[str]:
    """
    Finds all the alphabetical words in the input string (i.e. no numbers, punctuation, etc.)
    """
    # dont break up words with apostrophes, e.g. "don't" should be one word
    all_words = re.findall(r"[A-Za-z]+(?:['`][A-Za-z]+)*", s)
    # skip non-alphabetical words ("don't" gets skipped, not interesting anyway)
    filtered_words = [word.lower() for word in all_words if word.isalpha()]
    return filtered_words


def parse_solver_output(
    solver_output: str,
    message_words: list[str],
    words_prev_shown: set[str],
    distractor_added: Optional[DistractorSample] = None,
) -> dict[str, Any]:
    """
    Parses the string returned by the solver, determining whether a violation or
    mistake was made

    Args:
        solver_output: string returned by the solver
        message_words: list of words that were part of the input to the solver
        words_prev_shown: words already shown in previous turns as part of the
            base task
        distractor_added: dictionary containing the DistractorSample data if the message
            was a distractor question, else None

    Returns:
        violation_occurred: whether the solver output is a violation
        mistake_made: whether the solver output is a mistake. True if violation is True.
        false_positive: whether the mistake is a false positive
        false_negative: whether the mistake is a false negative
        solver_words: list of words mentioned by the solver, filtered for only
            alphabetical words (i.e. no numbers, punctuation, etc.)
    """
    solver_string = solver_output
    pattern = r"\[answer: ([^\]]+)\]"
    match = re.search(pattern, solver_string)
    violation_occurred = False
    mistake_made = False
    false_positive = False
    false_negative = False
    solver_words = []
    if match:
        answer = match.group(1).strip().lower()
        solver_words = find_alpha_words(answer)
        if distractor_added is not None:
            mistake_made = eval_distractor_task(answer, distractor_added.ideal)
        else:
            (
                violation_occurred,
                mistake_made,
                false_positive,
                false_negative,
            ) = eval_main_task(answer, message_words[0], words_prev_shown)
    else:
        violation_occurred = True
        mistake_made = True

    return {
        "violation_occurred": violation_occurred,
        "mistake_made": mistake_made,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "solver_words": solver_words,
    }


def eval_distractor_task(answer: str, ideal: str) -> bool:
    # whether a mistake is made
    return answer != ideal.strip().lower()


def eval_main_task(answer, curr_word, words_prev_shown):
    violation_occurred = False
    mistake_made = False
    false_positive = False
    false_negative = False
    if answer not in {"yes", "no"}:
        violation_occurred = True
        mistake_made = True
    elif answer == "no" and curr_word in words_prev_shown:
        false_negative = True
        mistake_made = True
    elif answer == "yes" and curr_word not in words_prev_shown:
        false_positive = True
        mistake_made = True

    return violation_occurred, mistake_made, false_positive, false_negative
