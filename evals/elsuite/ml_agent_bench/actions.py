import json
import re
from typing import Optional

from evals.elsuite.ml_agent_bench.high_level_actions import HIGH_LEVEL_ACTIONS
from evals.elsuite.ml_agent_bench.low_level_actions import LOW_LEVEL_ACTIONS
from evals.elsuite.ml_agent_bench.schema import Action

ACTION_SPACE = LOW_LEVEL_ACTIONS + HIGH_LEVEL_ACTIONS


def make_action_string(name: str, args: dict) -> str:
    stringified_args = json.dumps(args, indent=4)
    return f"Action: {name}\nAction Input: {stringified_args}"


def get_action(s: str) -> Optional[Action]:
    """Return an `Action` object from a string representation of an action, if it exists."""

    action_pattern = r"Action:\s*(.+)"
    args_pattern = r"Action Input:\s*(\{.*?\}|\S.*)"

    action_match = re.search(action_pattern, s)
    args_match = re.search(args_pattern, s, re.DOTALL)

    if not action_match:
        return None

    action_name = action_match.group(1).strip()
    action_args = None

    if args_match:
        args_str = args_match.group(1).strip()

        try:
            action_args = json.loads(args_str)
        except json.JSONDecodeError:
            action_args = args_str  # Return raw string if JSON parsing fails

    return Action(name=action_name, args=action_args)


def is_valid_action(action: Action) -> bool:
    """Return True if the action has a valid name and arguments, False otherwise."""

    assert isinstance(action, Action)

    if isinstance(action.args, str):
        return False

    for valid_action in ACTION_SPACE:
        if action.name != valid_action.name:
            continue

        actual_args = action.args.keys()
        expected_args = valid_action.usage.keys()

        return actual_args == expected_args

    return False
