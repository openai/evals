import logging
import re
import time
from typing import Optional

import docker
from beartype import beartype

from evals.elsuite.multistep_web_tasks.constants import ServiceIdentifier
from evals.elsuite.multistep_web_tasks.session import Session
from evals.elsuite.multistep_web_tasks.webarena.bash_env.actions import (
    BashAction,
    BashCommandAction,
    BashStopAction,
)
from evals.elsuite.multistep_web_tasks.webarena.bash_env.bash_utils import (
    BashEnvOutput,
    BashObservation,
)
from evals.elsuite.multistep_web_tasks.webarena.browser_env.actions import ActionParsingError
from evals.elsuite.multistep_web_tasks.webarena.core.env import LLMAgentEnv
from evals.elsuite.multistep_web_tasks.webarena.core.utils import BashExperimentConfig

logger = logging.getLogger(__name__)


class BashEnv(LLMAgentEnv):
    def __init__(
        self,
        session: Session,
        container_image: str = "dc-evals-bash",
        container_name: ServiceIdentifier = "bash",
    ):
        self.container_image = container_image
        self.container_name: ServiceIdentifier = container_name
        self.session = session

        self.container_wrapper = self._create_container_wrapper(self.session)

    @beartype
    def reset(self, experiment_config: Optional[BashExperimentConfig] = None) -> BashEnvOutput:
        """Output should be observation and info, but we don't have any info to output
        and the observation is just the output of the command"""
        # just make a new container
        self.container_wrapper.shutdown()
        # TODO: work out if there's a better way to wait on the container
        time.sleep(5)  # wait for the container to shut down
        self.container_wrapper = self._create_container_wrapper(self.session)

        if experiment_config is not None:
            self.setup(experiment_config)

        out = ""  # initial obs is empty
        env_output = BashEnvOutput(
            observation=BashObservation(output=out),
            reward=0.0,
            done=False,
            truncated=False,
            info=None,
        )
        return env_output

    def _create_container_wrapper(self, session: Session) -> "BashContainerWrapper":
        container_wrapper = BashContainerWrapper(
            session=session,
            name=self.container_name,
        )
        return container_wrapper

    def setup(self, experiment_config: BashExperimentConfig) -> None:
        """Run setup from the experiment config
        NOTE: we enable internet access for setup using the bridge network,
        since it's our own code and we need to install packages"""
        commands = experiment_config.setup_commands
        if commands is not None:
            bridge_network = self.session.docker_client.networks.get("bridge")
            bridge_network.connect(self.container_wrapper.container)  # type: ignore
            self.container_wrapper.run_commands(commands)
            bridge_network.disconnect(self.container_wrapper.container)  # type: ignore

    @beartype
    def step(self, action: BashAction) -> BashEnvOutput:
        """Output should be observation, reward, done, and info, but we don't have any info to output
        and the observation is just the output of the command"""
        if action.is_stop:
            assert isinstance(action, BashStopAction)
            return BashEnvOutput(
                observation=BashObservation(output=""),
                reward=0.0,
                done=True,
                truncated=False,
                info=None,
            )
        else:
            assert isinstance(action, BashCommandAction)
            out = self.container_wrapper.run_command(action.command)
            # obs, reward, terminated, truncated, info
            return BashEnvOutput(
                observation=BashObservation(output=out),
                reward=0.0,
                done=False,
                truncated=False,
                info=None,
            )

    def parse_action_string(self, action_string: str) -> BashAction:
        if action_string.startswith("stop"):
            match = re.search(r"stop ?\[(.+)\]", action_string)
            if not match:  # some tasks don't require an answer
                answer = ""
            else:
                answer = match.group(1)
            bash_action = BashStopAction(
                is_stop=True,
                raw_prediction=action_string,  # don't have access to raw
                parsed_prediction=action_string,
                answer=answer,
            )
            return bash_action
        elif action_string.startswith("bash"):
            match = re.search(r"bash ?\[(.+)\]", action_string)
            if not match:
                raise ActionParsingError("No command follows bash!")
            else:
                command = match.group(1)
            bash_action = BashCommandAction(
                is_stop=False,  # don't have access to raw
                raw_prediction=action_string,  # don't have access to raw
                parsed_prediction=action_string,
                command=command,
            )
            return bash_action
        else:
            logger.debug(f"Action '{action_string}' cannot be parsed as a BashAction")
            raise ActionParsingError(
                f"Action {action_string} not recognized as Bash command (must be prefixed with `stop` or `bash`)"
            )

    def close(self):
        self.container_wrapper.shutdown()


class BashContainerWrapper:
    def __init__(
        self,
        session: Session,
        name: ServiceIdentifier = "bash",
    ):
        self.active = True
        self.session = session
        # session handles building the image and container, then we register it
        self.container = session.setup_container(name)
        # TODO: work out if registering/attaching the container should happen inside `session.setup_container`
        self.session.register_container(name, self.container)
        self._setup(self.container)

    def _setup(self, container):
        if not self.active:
            raise Exception("BashContainerWrapper is not active!")

        # set up the current directory and environment variables
        try:
            container.exec_run(
                'bash -c "pwd > ~/.current_dir; declare -p > ~/.current_env_variables"'
            )
        except Exception as e:
            logger.error("BashContainerWrapper _setup failed!")
            raise e

    def run_command(self, command: str) -> str:
        if not self.active:
            raise Exception("BashContainerWrapper is not active!")

        wrapped_command = self._wrap_command(command)
        raw_out = self.container.exec_run(wrapped_command)  # type: ignore [docker type-hinting is bad]
        str_out = raw_out.output.decode("utf-8").strip()
        logger.info(f"\nCommand:\n{command}\n\nOutput:\n{str_out}\n=====\n")
        return str_out

    def run_commands(self, commands: list[str]) -> list[str]:
        outs = []
        for command in commands:
            out = self.run_command(command)
            outs.append(out)
        return outs

    def _wrap_command(self, command: str) -> str:
        # have to make sure all double quotes are properly escaped
        escaped_command = command.replace('"', '\\"')
        # command inspired by ARC Evals' approach, to handle statelessness of exec
        command_template = 'bash -c "cd $(cat ~/.current_dir) >/dev/null 2>&1; source ~/.current_env_variables 2> /dev/null && {command} 2>&1; pwd > ~/.current_dir; declare -p > ~/.current_env_variables"'
        return command_template.format(command=escaped_command)

    def shutdown(self):
        if not self.active:
            raise Exception("BashContainerWrapper is not active!")

        # Let the session handle container removal
        try:
            self.session.teardown_container(self.container.name)
            self.active = False
        except docker.errors.APIError as e:  # type: ignore
            logger.warning(f"Error shutting down container {self.container.name}: {e}")
