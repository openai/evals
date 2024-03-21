from dataclasses import dataclass

from beartype import beartype

from evals.elsuite.multistep_web_tasks.webarena.core.env import Action


@dataclass
class BashAction(Action):
    pass


@dataclass
class BashCommandAction(BashAction):
    command: str
    is_stop: bool


@dataclass
class BashStopAction(BashAction):
    answer: str
    is_stop: bool


@beartype
def bash_is_equivalent(a_action: BashAction, b_action: BashAction) -> bool:
    """Return True if two actions are equal.
    NOTE: this might not work great if formatting is slightly different
    but I think it's good enough"""
    return a_action.parsed_prediction == b_action.parsed_prediction
