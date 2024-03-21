import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

from evals.elsuite.multistep_web_tasks.webarena.core.env import DummyObservation, Observation
from evals.elsuite.multistep_web_tasks.webarena.core.utils import (
    BashBrowserExperimentConfig,
    BashExperimentConfig,
    BrowserExperimentConfig,
    ExperimentConfig,
)
from evals.task_state import TaskState

logger = logging.getLogger(__name__)


@dataclass
class MWTTaskState(TaskState):
    # NOTE: these defaults are just to appease dataclass
    # since we can't use kw_only until 3.10
    previous_action: str = "<ERROR: DEFAULT PREVIOUS_ACTION>"
    observation: Observation = DummyObservation()
    url: Optional[str] = None
    goal: str = "<ERROR: DEFAULT GOAL>"


def load_experiment_config_from_file(experiment_config_path: str) -> ExperimentConfig:
    with open(experiment_config_path) as f:
        experiment_config = json.load(f)
    return load_experiment_config_from_dict(experiment_config)


def load_experiment_config_from_dict(experiment_config_dict: dict[str, Any]) -> ExperimentConfig:
    if experiment_config_dict["env_type"] == "browser":
        experiment_config = BrowserExperimentConfig.from_dict(experiment_config_dict)
    elif experiment_config_dict["env_type"] == "bash":
        experiment_config = BashExperimentConfig.from_dict(experiment_config_dict)
    elif experiment_config_dict["env_type"] == "bash_browser":
        experiment_config = BashBrowserExperimentConfig.from_dict(experiment_config_dict)
    else:
        raise ValueError(f"Unknown experiment config type {experiment_config_dict['type']}")
    return experiment_config
