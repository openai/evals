import pytest

from evals.elsuite.already_said_that.distractors import (
    proc_distractors_ambiguous_sentences,
    proc_distractors_first_letters,
    proc_distractors_reverse_sort_words_eng,
    proc_which_is_heavier,
)


@pytest.fixture
def which_is_heavier_samples():
    return [
        {
            "input": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "Is 2 kilograms of eggshells heavier than 1 kilogram of gold bars? Answer Yes or No",
                },
            ],
            "ideal": "Yes",
        },
        {
            "input": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "Is 10 kilograms of granite slabs heavier than 20 kilograms of balloons? Answer Yes or No",
                },
            ],
            "ideal": "No",
        },
    ]


@pytest.fixture
def first_letters_samples():
    return [
        {
            "input": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Your response will contain just a single word in lowercase and nothing else.",
                },
                {
                    "role": "user",
                    "content": 'What is the word obtained from concatenating the first letters of the words in the following sentence: "Dear Elena find irregular new energies."?',
                },
            ],
            "ideal": "define",
        },
        {
            "input": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Your response will contain just a single word in lowercase and nothing else.",
                },
                {
                    "role": "user",
                    "content": 'What is the word obtained from concatenating the first letters of the words in the following sentence: "Bring roses on a doorstep."?',
                },
            ],
            "ideal": "broad",
        },
    ]


@pytest.fixture
def ambiguous_sentences_samples():
    return [
        {
            "input": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "This morning, Joey built a sand castle on the beach, and put a toy  flag in the highest tower, but this afternoon the tide knocked it down.. What did the the tide knock down?",
                },
            ],
            "ideal": "castle",
        },
        {
            "input": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "Sam took French classes from Adam, because he was eager to speak it fluently.. Who was eager to speak French fluently?",
                },
            ],
            "ideal": "Sam",
        },
    ]


@pytest.fixture
def reverse_sort_words_eng_samples():
    return [
        {
            "input": [
                {
                    "role": "system",
                    "content": "Sort the following comma-separated words in reversed alphabetical order (respond as concisely as possible and only include the comma-separated words in your response):",
                },
                {
                    "role": "user",
                    "content": "gargoyle, porcupine, whirlwind, zest, lily, molasses, penguin, satchel, igloo, viper",
                },
            ],
            "ideal": "zest, whirlwind, viper, satchel, porcupine, penguin, molasses, lily, igloo, gargoyle",
        },
        {
            "input": [
                {
                    "role": "system",
                    "content": "Sort the following comma-separated words in reversed alphabetical order (respond as concisely as possible and only include the comma-separated words in your response):",
                },
                {
                    "role": "user",
                    "content": "marigold, opal, labyrinth, silhouette, whirlpool, trumpet, forge, quill, knapsack, emblem",
                },
            ],
            "ideal": "whirlpool, trumpet, silhouette, quill, opal, marigold, labyrinth, knapsack, forge, emblem",
        },
    ]


def test_proc_distractors_which_is_heavier(which_is_heavier_samples):
    result = proc_which_is_heavier(which_is_heavier_samples)
    assert len(result) == 2
    assert result[0].question == "Is 2 kilograms of eggshells heavier than 1 kilogram of gold bars?"
    assert result[0].ideal == "yes"
    assert (
        result[1].question
        == "Is 10 kilograms of granite slabs heavier than 20 kilograms of balloons?"
    )
    assert result[1].ideal == "no"


def test_proc_distractors_first_letter(first_letters_samples):
    result = proc_distractors_first_letters(first_letters_samples)
    assert len(result) == 2
    assert (
        result[0].question
        == 'What is the word obtained from concatenating the first letters of the words in the following sentence: "Dear Elena find irregular new energies."?'
    )
    assert result[0].ideal == "define"
    assert (
        result[1].question
        == 'What is the word obtained from concatenating the first letters of the words in the following sentence: "Bring roses on a doorstep."?'
    )
    assert result[1].ideal == "broad"


def test_proc_distractors_ambiguous_sentences(ambiguous_sentences_samples):
    result = proc_distractors_ambiguous_sentences(ambiguous_sentences_samples)
    assert len(result) == 2
    assert (
        result[0].question
        == "This morning, Joey built a sand castle on the beach, and put a toy  flag in the highest tower, but this afternoon the tide knocked it down.. What did the the tide knock down?"
    )
    assert result[0].ideal == "castle"
    assert (
        result[1].question
        == "Sam took French classes from Adam, because he was eager to speak it fluently.. Who was eager to speak French fluently?"
    )
    assert result[1].ideal == "sam"


def test_proc_distractors_reverse_sort_words_eng(reverse_sort_words_eng_samples):
    result = proc_distractors_reverse_sort_words_eng(reverse_sort_words_eng_samples)
    assert len(result) == 2
    assert (
        result[0].question
        == "Sort the following comma-separated words in reversed alphabetical order: gargoyle, porcupine, whirlwind, zest, lily, molasses, penguin, satchel, igloo, viper"
    )
    assert (
        result[0].ideal
        == "zest, whirlwind, viper, satchel, porcupine, penguin, molasses, lily, igloo, gargoyle"
    )
    assert (
        result[1].question
        == "Sort the following comma-separated words in reversed alphabetical order: marigold, opal, labyrinth, silhouette, whirlpool, trumpet, forge, quill, knapsack, emblem"
    )
    assert (
        result[1].ideal
        == "whirlpool, trumpet, silhouette, quill, opal, marigold, labyrinth, knapsack, forge, emblem"
    )
