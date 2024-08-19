"""Script for running all the docker containers for testing purposes"""

import logging

import docker

from evals.elsuite.multistep_web_tasks.session import Session
from evals.elsuite.multistep_web_tasks.utils import (
    BashBrowserExperimentConfig,
    load_experiment_config_from_file,
)
from evals.elsuite.multistep_web_tasks.webarena.bash_env.basic_bash_env import BashEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    session = Session(docker.from_env())
    session.containers_to_setup = {
        "homepage",
        "shopping",
        "shopping-admin",
        "reddit",
        "wikipedia",
        "flask-playwright",
    }
    # session.containers_to_setup = {"flask-playwright", "wikipedia", "reddit", "shopping"}
    with session:
        experiment_config = load_experiment_config_from_file(
            "/datadrive/code/dangerous-capability-evaluations/evals/registry/data/multistep-web-tasks/task_7.jsonl"
        )
        assert isinstance(experiment_config, BashBrowserExperimentConfig)
        bash_config, browser_config = experiment_config.to_separate_configs()
        bash_env = BashEnv(session, container_name="bash")
        bash_env.reset(bash_config)
        input("Containers running! Press enter to exit.")
