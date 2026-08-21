from typing import Optional

from pytest import mark

from evals.elsuite.mmmu.eval import _extract_multiple_choice_answer


@mark.parametrize(
    "text, expected",
    [
        ("ANSWER: A", "A"),
        ("Reasoning first.\nANSWER: B.", "B"),
        ("ANSWER: AB", None),
        ("ANSWER: A1", None),
        ("ANSWER: A\nActually, ANSWER: B", "B"),
        ("No explicit answer", None),
    ],
)
def test_extract_multiple_choice_answer_uses_final_valid_marker(
    text: str, expected: Optional[str]
) -> None:
    assert _extract_multiple_choice_answer(text) == expected
