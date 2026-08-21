from copy import deepcopy

from pytest import mark

from evals.elsuite.utils import fuzzy_match, normalize, scrub_formatting_from_prompt


@mark.parametrize(
    "s, expected",
    [
        ("", ""),
        ("Hello", "hello"),
        ("hello\nworld", "hello world"),
    ],
)
def test_normalize(s: str, expected: str):
    assert normalize(s) == expected


@mark.parametrize(
    "s1, s2, expected",
    [
        ("", "", True),
        ("x", "", False),
        ("Hello", "Hello", True),
        ("hello", "othello", True),
        ("hello", "oh tello", False),
        ("Hello World", "foo\nhello world", True),
        ("who's there?", "whos there", True),
        ("who's there?", "whosthere", False),
        ("an apple a day that the", "apple day that", True),
    ],
)
def test_fuzzy_match(s1: str, s2: str, expected: bool):
    assert fuzzy_match(s1, s2) == expected
    assert fuzzy_match(s2, s1) == expected


def test_scrub_formatting_from_chat_prompt_does_not_mutate_input():
    prompt = [
        {"role": "system", "content": "Use {name} exactly", "name": "instruction"},
        {"role": "user", "content": "Value: {value}"},
    ]
    original = deepcopy(prompt)

    scrubbed = scrub_formatting_from_prompt(prompt)

    assert prompt == original
    assert scrubbed == [
        {"role": "system", "content": "Use {{name}} exactly", "name": "instruction"},
        {"role": "user", "content": "Value: {{value}}"},
    ]
    assert scrubbed is not prompt
    assert scrubbed[0] is not prompt[0]
    assert scrubbed[1] is not prompt[1]


def test_scrub_formatting_from_chat_prompt_is_stable_across_repeated_calls():
    prompt = [{"role": "user", "content": "Return {answer}"}]

    first = scrub_formatting_from_prompt(prompt)
    second = scrub_formatting_from_prompt(prompt)

    assert first == second == [{"role": "user", "content": "Return {{answer}}"}]
    assert prompt == [{"role": "user", "content": "Return {answer}"}]


def test_scrub_formatting_from_string_prompt_preserves_behavior():
    assert scrub_formatting_from_prompt("Return {answer}") == "Return {{answer}}"
