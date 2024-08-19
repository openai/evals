import logging
from typing import Optional, Union

from beartype import beartype
from playwright.sync_api import ViewportSize

from evals.elsuite.multistep_web_tasks.session import Session
from evals.elsuite.multistep_web_tasks.webarena.bash_browser_env.bash_browser_utils import (
    BashBrowserEnvOutput,
)
from evals.elsuite.multistep_web_tasks.webarena.bash_env.actions import BashAction
from evals.elsuite.multistep_web_tasks.webarena.bash_env.basic_bash_env import BashEnv
from evals.elsuite.multistep_web_tasks.webarena.browser_env.actions import (
    ActionParsingError,
    BrowserAction,
)
from evals.elsuite.multistep_web_tasks.webarena.browser_env.basic_browser_env import BrowserEnv
from evals.elsuite.multistep_web_tasks.webarena.core.env import LLMAgentEnv
from evals.elsuite.multistep_web_tasks.webarena.core.utils import BashBrowserExperimentConfig

logger = logging.getLogger(__name__)


class BashBrowserEnv(LLMAgentEnv):
    """Currently, this is implemented as a wrapper around a BashEnv and a
    BrowserEnv.  I'm not sure if this is ideal -- I'm worried that e.g. running
    a bash command that should change something for the BrowserEnv won't
    actually register that change, but I think since they're both talking to the
    same underlying Docker containers, it should be okay."""

    def __init__(
        self,
        # bash specific
        session: Session,
        container_image: str = "dc-evals-bash",
        container_name: str = "bash",
        # browser specific
        max_page_length: int = 8192,
        headless: bool = True,
        slow_mo: int = 0,
        observation_type: str = "html",
        current_viewport_only: bool = False,
        viewport_size: ViewportSize = {"width": 1280, "height": 720},
        save_trace_enabled: bool = False,
        sleep_after_execution: float = 0.5,
    ):
        self.bash_env = BashEnv(
            container_image=container_image,
            container_name=container_name,
            session=session,
        )
        self.browser_env = BrowserEnv(
            session=session,
            max_page_length=max_page_length,
            headless=headless,
            slow_mo=slow_mo,
            observation_type=observation_type,
            current_viewport_only=current_viewport_only,
            viewport_size=viewport_size,
            save_trace_enabled=save_trace_enabled,
            sleep_after_execution=sleep_after_execution,
        )

    @property
    def page(self):
        return self.browser_env.page

    @beartype
    def reset(
        self, experiment_config: Optional[BashBrowserExperimentConfig] = None
    ) -> BashBrowserEnvOutput:
        """Reset both the bash env and the browser env.
        TODO: work out what observation to return
        - for now, returning the browser output"""
        if experiment_config is None:
            self.bash_env.reset()
            browser_output = self.browser_env.reset()
        else:
            bash_config, browser_config = experiment_config.to_separate_configs()
            self.bash_env.reset(experiment_config=bash_config)
            browser_output = self.browser_env.reset(experiment_config=browser_config)
        return browser_output

    def setup(self, experiment_config: BashBrowserExperimentConfig) -> None:
        """Run setup from the experiment config"""
        bash_config, browser_config = experiment_config.to_separate_configs()
        self.bash_env.setup(bash_config)
        self.browser_env.setup(browser_config)

    @beartype
    def step(self, action: Union[BashAction, BrowserAction]) -> BashBrowserEnvOutput:
        if isinstance(action, BashAction):
            env_output = self.bash_env.step(action)
        elif isinstance(action, BrowserAction):
            env_output = self.browser_env.step(action)
        else:
            raise ValueError(f"Action {action} is not a BashAction or BrowserAction!")
        return env_output

    def parse_action_string(self, action_string: str) -> Union[BashAction, BrowserAction]:
        """Parse an action string into a BashAction or BrowserAction
        TODO: work out if it matters that stop actions will always be parsed
        as bash actions -- should I have a separate StopAction class?"""
        try:
            action = self.bash_env.parse_action_string(action_string)
        except ActionParsingError:
            try:
                action = self.browser_env.parse_action_string(action_string)
            except ActionParsingError:
                raise ActionParsingError(f"Could not parse {action_string} as bash or browser")
        return action

    def close(self):
        self.bash_env.close()
        self.browser_env.close()
