from pytest import mark
from evals.elsuite.utils import fuzzy_match, normalize

@mark.parametrize("s, expected", [
    ("", ""),
    ("Hello", "hello"),
    ("hello\nworld", "hello world"),
])
def test_normalize(s: str, expected: str):
    assert normalize(s) == expected

@mark.parametrize("s1, s2, expected", [
    ("", "", True),
    ("x", "", False),
    ("Hello", "Hello", True),
    ("hello", "othello", True),
    ("hello", "oh tello", False),
    ("Hello World", "foo\nhello world", True),
    ("who's there?", "whos there", True),
    ("who's there?", "whosthere", False),
    ("an apple a day that the", "apple day that", True),
])
def test_fuzzy_match(s1: str, s2: str, expected: bool):
    assert fuzzy_match(s1, s2) == expected
    assert fuzzy_match(s2, s1) == expected
