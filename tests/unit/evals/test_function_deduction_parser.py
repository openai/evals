from typing import Any

import pytest


@pytest.mark.parametrize(
    "response, expected",
    [
        ("1,2,3", (1, 2, 3)),
        ("[1,2,3]", (1, 2, 3)),
        ('"1","2","3"', (1, 2, 3)),
        ("-1,-2,-3", (-1, -2, -3)),
        ("42", (42,)),
    ],
)
def test_function_deduction_parser_preserves_integer_separators(
    response: str, expected: tuple[int, ...], monkeypatch: Any
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    from evals.elsuite.function_deduction.eval import FunctionDeductionEval

    evaluator = object.__new__(FunctionDeductionEval)

    assert evaluator._parse_raw_response(response) == expected
