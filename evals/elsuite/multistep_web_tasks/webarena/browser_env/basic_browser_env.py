import logging
import time
from pathlib import Path
from typing import Optional, Union

from beartype import beartype

from evals.elsuite.multistep_web_tasks.session import Session
from evals.elsuite.multistep_web_tasks.webarena.browser_env.actions import (
    BrowserAction,
    BrowserActionTypes,
    create_id_based_action,
    execute_action,
)
from evals.elsuite.multistep_web_tasks.webarena.browser_env.browser_utils import (
    BrowserEnvOutput,
    BrowserObservation,
    DetachedPage,
    PageInfo,
)
from evals.elsuite.multistep_web_tasks.webarena.browser_env.processors import (
    ObservationHandler,
    ObservationMetadata,
)
from evals.elsuite.multistep_web_tasks.webarena.core.env import LLMAgentEnv
from evals.elsuite.multistep_web_tasks.webarena.core.playwright_api import (
    ClientForwarder,
    PageForwarder,
    ViewportSize,
)
from evals.elsuite.multistep_web_tasks.webarena.core.utils import BrowserExperimentConfig

logger = logging.getLogger(__name__)


class BrowserEnv(LLMAgentEnv):
    """This class is a rewrite of ScriptBrowserEnv from WebArena that fits
    with my refactor of the codebase.

    From the original:
    "The goal of this environment is to produce a prototype of a browser environment.
    In the end, we want to support a fully configurable browser environment with wide
    range of action spaces and observation spaces, both structured and unstructured.
    But in this prototype, we just support action space specified by Playwright script,
    and observation space is the html content of the page."
    """

    @beartype
    def __init__(
        self,
        session: Session,
        max_page_length: int = 8192,
        headless: bool = True,
        slow_mo: int = 0,
        observation_type: str = "html",
        current_viewport_only: bool = False,
        viewport_size: ViewportSize = {"width": 1280, "height": 720},
        save_trace_enabled: bool = False,
        sleep_after_execution: float = 0.5,
    ):
        self.session = session
        # TODO: make Space[Action] = ActionSpace
        # self.action_space = get_action_space()  # type: ignore[assignment]
        self.headless = headless
        self.slow_mo = slow_mo
        self.current_viewport_only = current_viewport_only
        self.reset_finished = False
        self.viewport_size = viewport_size
        self.save_trace_enabled = save_trace_enabled
        self.sleep_after_execution = sleep_after_execution

        if observation_type in ["html", "accessibility_tree"]:
            self.text_observation_type = observation_type
            self.image_observation_type = ""
            self.main_observation_type = "text"
        elif observation_type == "image":
            self.image_observation_type = observation_type
            self.text_observation_type = ""  # type: ignore[assignment]
            self.main_observation_type = "image"
        else:
            raise ValueError(f"Unsupported observation type: {observation_type}")

        self.observation_handler = ObservationHandler(
            self.main_observation_type,
            self.text_observation_type,
            self.image_observation_type,
            self.current_viewport_only,
            self.viewport_size,
        )

    def sync_playwright_api(self, experiment_config: BrowserExperimentConfig) -> PageForwarder:
        """Possible TODO: move the setup logic from the API to this function so
        that we can control it from the client"""
        # session handles building the image and container, then we register it
        API_CONTAINER_NAME = "flask-playwright"
        self.container = self.session.setup_container(API_CONTAINER_NAME)
        # TODO: work out if registering/attaching the container should happen inside `session.setup_container`
        self.session.register_container(API_CONTAINER_NAME, self.container)
        viewport_size: ViewportSize = {
            "width": experiment_config.viewport_width,
            "height": experiment_config.viewport_height,
        }
        page = PageForwarder(self.container, viewport_size)
        # wait here for the page container to be ready
        # TODO: work out if this should happen in PageForwarder.__init__ or here or .setup()
        logger.info(f"Waiting for container '{self.container.name}' to be ready...")
        self.session._is_container_ready(self.container.name)
        logger.info(f"Container '{self.container.name}' is ready.")
        page.setup()
        return page

    def setup(self, experiment_config: BrowserExperimentConfig) -> None:
        """NOTE: we diverge from WebArena here, and use the flask-playwright API I (Ian) made.
        It is set up and managed in sync_playwright_api"""
        self.page = self.sync_playwright_api(experiment_config)
        start_url = experiment_config.start_url
        self.page.goto(start_url)

    def parse_action_string(self, action_string: str) -> BrowserAction:
        action = create_id_based_action(action_string)
        action.raw_prediction = action_string  # We don't have access to raw pred anymore
        action.parsed_prediction = action_string
        return action

    @beartype
    def get_page_client(self, page: PageForwarder) -> ClientForwarder:
        return page.client  # type: ignore

    @beartype
    def _get_obs(self) -> BrowserObservation:
        obs = self.observation_handler.get_observation(self.page, self.get_page_client(self.page))
        return obs

    @beartype
    def _get_obs_metadata(self) -> dict[str, ObservationMetadata]:
        metadata = self.observation_handler.get_observation_metadata()
        return metadata

    @beartype
    def reset(
        self,
        *,
        experiment_config: Optional[BrowserExperimentConfig] = None,
        seed: Optional[int] = None,
        options: Optional[dict[str, str]] = None,
    ) -> BrowserEnvOutput:
        """
        Reset the environment.
        :param options: options for the environment. The current supported options are:
            - "storage_state": the storage state of the browser. It is a file path to a json file.
        """
        # TODO: work out if we should be resetting through to the Gym base class
        # super().reset(seed=seed, options=options)
        # TODO: clean up the container and reuse it rather than tearing down and making a new one
        if self.reset_finished:
            self.page.shutdown()
            self.session.teardown_container(self.page.container.name)
        if experiment_config is not None:
            self.setup(experiment_config)
        self.reset_finished = True

        if self.sleep_after_execution > 0:
            time.sleep(self.sleep_after_execution)

        observation = self._get_obs()
        observation_metadata = self._get_obs_metadata()
        info = PageInfo(
            page=DetachedPage(self.page.url, ""),
            fail_error="",
            observation_metadata=observation_metadata,
        )

        env_output = BrowserEnvOutput(
            observation=observation,
            reward=0.0,
            done=False,
            truncated=False,
            info=info,
        )

        return env_output

    @beartype
    def save_trace(self, trace_path: Union[str, Path]) -> None:
        raise NotImplementedError("TODO: traces with flask-playwright api")

    @beartype
    def close(self) -> None:
        if self.reset_finished:
            self.page.shutdown()
            self.session.teardown_container(self.page.container.name)

    def step(self, action: BrowserAction) -> BrowserEnvOutput:
        if not self.reset_finished:
            raise RuntimeError("Call reset first before calling step.")

        # handle stop separately
        if action.data["action_type"] == BrowserActionTypes.STOP:
            return BrowserEnvOutput(
                observation=self._get_obs(),
                reward=0.0,
                done=True,
                truncated=False,
                info=PageInfo(
                    page=DetachedPage(self.page.url, self.page.content()),
                    fail_error="",
                    observation_metadata=self._get_obs_metadata(),
                ),
            )

        success = False
        fail_error = ""
        previous_obs = self._get_obs()
        try:
            execute_action(
                action,
                self.page,
                self.observation_handler.action_processor,
            )
            success = True
        except Exception as e:
            logger.warning(f"Failed to execute action {action}: {e}")
            fail_error = str(e)

        # hard sleep TODO[shuyanzh] suboptimal, may need to check network
        if self.sleep_after_execution > 0:
            time.sleep(self.sleep_after_execution)

        observation = self._get_obs()
        if observation.data == previous_obs.data:
            logger.warning(
                f"\nObservation did not change after executing action:\n{action}\n=====\n"
            )
        observation_metadata = self._get_obs_metadata()

        info = PageInfo(
            page=DetachedPage(self.page.url, self.page.content()),
            fail_error=fail_error,
            observation_metadata=observation_metadata,
        )
        env_output = BrowserEnvOutput(
            observation=observation,
            reward=float(success),
            done=False,
            truncated=False,
            info=info,
        )
        return env_output
