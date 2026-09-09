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


@mark.parametrize("prompt", ["No braces", 'Return {"answer": 42}', "{{already escaped}}"])
def test_scrub_formatting_from_text_prompt(prompt):
    scrubbed = scrub_formatting_from_prompt(prompt)
    assert scrubbed.format() == prompt


def test_scrub_formatting_does_not_modify_chat_messages():
    prompt = [
        {"role": "system", "content": 'Return {"answer": 42}'},
        {"role": "user", "content": "Evaluate {expression}", "name": "caller"},
        {"role": "assistant", "function_call": {"name": "answer", "arguments": "{}"}},
    ]
    original = [dict(message) for message in prompt]

    first = scrub_formatting_from_prompt(prompt)
    second = scrub_formatting_from_prompt(prompt)

    assert prompt == original
    assert first == second
    assert first[0]["content"] == 'Return {{"answer": 42}}'
    assert first[1]["content"] == "Evaluate {{expression}}"
    assert first[1]["name"] == "caller"
    assert first[2] == original[2]
    assert first[0]["content"].format() == original[0]["content"]
