"""This file contains abstract classes representing Actions, Observations, and Environments.
This abstraction should be able to handle ScriptBrowserEnv and BashEnv, as well as a combination
of the two."""

from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
from typing import Any, NamedTuple, Optional

from gymnasium import Env

from evals.elsuite.multistep_web_tasks.webarena.core.utils import ExperimentConfig

# These classes are extremely small,
# and are just there for type-checking really
# TODO: work out if these should be implemented differently -
# traits, mixins, interfaces?


@dataclass
class Action(ABC):
    """Since we're always working with LMs, there will always be a
    raw text prediction. Additionally, the environment doesn't decide when to stop,
    the agent does. (this distinction is a little messy)"""

    raw_prediction: str
    parsed_prediction: str
    is_stop: bool


class ParsingErrorAction(Action):
    """This is a special action that is returned when the agent's prediction fails to be parsed
    properly"""

    parsed_prediction: str = "ERROR: Failed to parse action. Make sure to wrap the arguments inside [] and the whole action inside ```. Visit the homepage for available sites."


class Observation(ABC):
    @abstractproperty
    def data(self) -> Any:
        """This property is the main way to get the actual contents of
        an observation."""
        raise NotImplementedError


class DummyObservation(Observation):
    def data(self) -> Any:
        return "<ERROR: DUMMY OBSERVATION>"


class Info(ABC):
    pass


@dataclass
class EnvOutput:
    """All environments should output a 5-tuple
    TODO: work out if truncated and info are strictly necessary"""

    observation: Observation
    reward: float
    done: bool
    truncated: bool = False
    info: Optional[Info] = None


class TrajectoryStep(NamedTuple):
    action: Optional[Action]
    env_output: EnvOutput


class Trajectory(list[TrajectoryStep]):
    """Not sure if subclassing list here is a wise choice"""

    def __init__(self, iterable: list[TrajectoryStep]):
        assert all(isinstance(x, TrajectoryStep) for x in iterable)
        super().__init__(iterable)

    def pretty_string(self) -> str:
        """TODO: improve the way this string is built"""
        s = "================================\n"
        s += "========== Trajectory ==========\n"
        s += "================================\n"
        for i, item in enumerate(self):
            s += f"========== Step {i} ==========\n"
            if item.action is None:
                s += "Action: None\n"
                s += "-----------\n\n"
            else:
                s += f"Raw action:\n{item.action.raw_prediction}\n"
                s += f"Parsed action:\n{item.action.parsed_prediction}\n"
                s += "-----------\n\n"
            s += f"Observation:\n{item.env_output.observation.data}\n\n"
        return s


class LLMAgentEnv(ABC, Env[Observation, Action]):
    """Abstract subclass of gym's Env class for LLM agents to interact with.
    Not sure if this intermediate is necessary or we could just go straight to
    BashEnv, ScriptBrowserEnv, etc."""

    @abstractmethod
    def reset(
        self,
        *,
        experiment_config: Optional[ExperimentConfig] = None,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> EnvOutput:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Action) -> EnvOutput:
        raise NotImplementedError

    @abstractmethod
    def parse_action_string(self, action_string: str) -> Action:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


@dataclass
class ExperimentResult(ABC):
    score: float
    trajectory: Trajectory
    env: LLMAgentEnv
    experiment_config: ExperimentConfig
