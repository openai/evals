import pytest

from evals.elsuite.ml_agent_bench.actions import (
    ACTION_SPACE,
    get_action,
    is_valid_action,
    make_action_string,
)
from evals.elsuite.ml_agent_bench.schema import Action


def test_make_action_string():
    # Given
    name = "name"
    args = {"arg": "value"}
    expected = """
Action: name
Action Input: {
    "arg": "value"
}""".strip()

    # When
    actual = make_action_string(name, args)

    # Then
    assert actual == expected, f"Expected: {expected}, Actual: {actual}"


def test_empty_string():
    # Given
    input_str = ""

    # When
    actual = get_action(input_str)

    # Then
    assert actual is None


def test_missing_curly_braces():
    # Given
    input_str = """
Action: MissingBraces
Action Input: 
    "arg1": "value1"
"""
    args_str = input_str.strip().split("Action Input: ")[1].strip()
    expected = Action("MissingBraces", args_str)

    # When
    actual = get_action(input_str)

    # Then
    assert actual.name == expected.name
    assert actual.args == expected.args


def test_args_on_multiple_lines():
    # Given
    input_str = """
Action: Valid Name
Action Input: {
    "arg1": "value1",
    "arg2": "value2"
}
"""
    expected = Action("Valid Name", {"arg1": "value1", "arg2": "value2"})

    # When
    actual = get_action(input_str)

    # Then
    assert actual.name == expected.name
    assert actual.args == expected.args


def test_args_on_single_line():
    # Given
    input_str = """
Action: Valid Name
Action Input: {"arg1": "value1", "arg2": "value2"}
"""
    expected = Action("Valid Name", {"arg1": "value1", "arg2": "value2"})

    # When
    actual = get_action(input_str)

    # Then
    assert actual.name == expected.name
    assert actual.args == expected.args


def test_special_characters_in_name():
    # Given
    input_str = """
Action: Special!@#Name
Action Input: {
    "arg1": "value1"
}
"""
    expected = Action("Special!@#Name", {"arg1": "value1"})

    # When
    actual = get_action(input_str)

    # Then
    assert actual.name == expected.name
    assert actual.args == expected.args


def test_invalid_arguments():
    # Given
    input_str = """
Action: Invalid Arguments
Action Input: "some invalid json string"
"""
    expected = Action("Invalid Arguments", "some invalid json string")

    # When
    actual = get_action(input_str)

    # Then
    assert actual.name == expected.name
    assert actual.args == expected.args


def test_surrounded_by_additional_text():
    # Given
    input_str = """
Some thoughts about which action to take.

Action: Edit Script (AI)
Action Input: {
    "script_name": "improved_train.py",
    "edit_instruction": "Correct the line that initializes the q_table.",
    "save_name": "improved_train.py"
}

Please execute that action.
"""
    expected = Action(
        name="Edit Script (AI)",
        args={
            "script_name": "improved_train.py",
            "edit_instruction": "Correct the line that initializes the q_table.",
            "save_name": "improved_train.py",
        },
    )

    # When
    actual = get_action(input_str)

    # Then
    assert actual.name == expected.name
    assert actual.args == expected.args


@pytest.mark.parametrize("action_info", ACTION_SPACE)
def test_is_valid_action_with_correct_args(action_info):
    action = Action(
        name=action_info.name,
        args={k: "test_value" for k in action_info.usage.keys()},
    )

    assert is_valid_action(action)


@pytest.mark.parametrize("action_info", ACTION_SPACE)
def test_is_valid_action_with_incorrect_args(action_info):
    incorrect_args = {k + "_wrong": "test_value" for k in action_info.usage.keys()}
    action = Action(name=action_info.name, args=incorrect_args)

    assert not is_valid_action(action)


@pytest.mark.parametrize("action_info", ACTION_SPACE)
def test_is_valid_action_with_missing_args(action_info):
    if action_info.usage.keys():
        new_keys = list(action_info.usage.keys())[:-1]  # remove one arg if possible
        missing_args = {k: "test_value" for k in new_keys}
        action = Action(name=action_info.name, args=missing_args)

        assert not is_valid_action(action)
    else:
        pytest.skip("Action does not have any args to test for missing scenario.")
