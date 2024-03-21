"""Script to run end-to-end evaluation on the benchmark"""
import argparse
import logging
from pathlib import Path

from evals.elsuite.multistep_web_tasks.session import Session
from evals.elsuite.multistep_web_tasks.utils import MWTTaskState
from evals.elsuite.multistep_web_tasks.webarena.bash_browser_env.bash_browser_env import (
    BashBrowserEnv,
)
from evals.elsuite.multistep_web_tasks.webarena.bash_env.bash_utils import BashEnvOutput
from evals.elsuite.multistep_web_tasks.webarena.bash_env.basic_bash_env import BashEnv
from evals.elsuite.multistep_web_tasks.webarena.browser_env.actions import (
    ActionParsingError,
    is_equivalent,
)
from evals.elsuite.multistep_web_tasks.webarena.browser_env.basic_browser_env import BrowserEnv
from evals.elsuite.multistep_web_tasks.webarena.browser_env.browser_utils import BrowserEnvOutput
from evals.elsuite.multistep_web_tasks.webarena.core.env import (
    ExperimentResult,
    LLMAgentEnv,
    ParsingErrorAction,
    Trajectory,
    TrajectoryStep,
)
from evals.elsuite.multistep_web_tasks.webarena.core.utils import (
    BashBrowserExperimentConfig,
    BashExperimentConfig,
    BrowserExperimentConfig,
    EarlyStopConfig,
    ExperimentConfig,
)
from evals.elsuite.multistep_web_tasks.webarena.evaluation_harness.evaluators import (
    evaluator_router,
)
from evals.elsuite.multistep_web_tasks.webarena.task_description import (
    DEFAULT_TASK_DESCRIPTION_TEMPLATE,
)
from evals.solvers.solver import Solver, SolverResult
from evals.task_state import Message

logger = logging.getLogger(__name__)


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end evaluation on the benchmark")
    parser.add_argument("--render", action="store_true", help="Render the browser")
    parser.add_argument(
        "--slow_mo",
        type=int,
        default=0,
        help="Slow down the browser by the specified amount",
    )
    parser.add_argument("--action_set_tag", default="id_accessibility_tree", help="Action type")
    parser.add_argument(
        "--observation_type",
        choices=["accessibility_tree", "html", "image"],
        default="accessibility_tree",
        help="Observation type",
    )
    parser.add_argument(
        "--current_viewport_only",
        action="store_true",
        help="Only use the current viewport for the observation",
    )
    parser.add_argument("--viewport_width", type=int, default=1280)
    parser.add_argument("--viewport_height", type=int, default=720)
    parser.add_argument("--save_trace_enabled", action="store_true")
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)

    parser.add_argument("--max_steps", type=int, default=30)

    # agent config
    parser.add_argument("--agent_type", type=str, default="prompt")
    parser.add_argument(
        "--instruction_path",
        type=str,
        default="agents/prompts/state_action_agent.json",
    )
    parser.add_argument(
        "--parsing_failure_th",
        help="When concesecutive parsing failure exceeds this threshold, the agent will stop",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--repeating_action_failure_th",
        help="When concesecutive repeating action exceeds this threshold, the agent will stop",
        type=int,
        default=3,
    )

    # example config
    parser.add_argument("--test_start_idx", type=int, default=0)
    parser.add_argument("--test_end_idx", type=int, default=1000)

    args = parser.parse_args()

    # check the whether the action space is compatible with the observation space
    if (
        args.action_set_tag == "id_accessibility_tree"
        and args.observation_type != "accessibility_tree"
    ):
        raise ValueError(
            f"Action type {args.action_set_tag} is incompatible with the observation type {args.observation_type}"
        )

    return args


def run_experiment(
    agent: Solver,
    experiment_config: ExperimentConfig,
    session: Session,
) -> ExperimentResult:
    logger.info(f"\nExperiment config:\n{experiment_config}\n=====\n")
    env = setup_env(experiment_config, session)
    # TODO: maybe make this config editable or put inside generate trajectory
    early_stop_config = EarlyStopConfig()
    trajectory = generate_trajectory(agent, env, experiment_config, early_stop_config)
    result = evaluate_trajectory(trajectory, env, experiment_config)
    env.close()
    logger.info(f"\nResult score: {result.score}\n=====\n")
    return result


def generate_trajectory(
    agent: Solver,
    env: LLMAgentEnv,
    experiment_config: ExperimentConfig,
    early_stop_config: EarlyStopConfig,
) -> Trajectory:
    env_output = env.reset(experiment_config=experiment_config)
    logger.info(f"\nInitial observation:\n{env_output.observation}\n=====\n")
    trajectory: Trajectory = Trajectory([TrajectoryStep(None, env_output)])
    while _episode_should_continue(trajectory, early_stop_config):
        task_state = _task_state_from_trajectory(trajectory, experiment_config.goal)
        solver_output = agent(task_state)
        assert isinstance(solver_output, SolverResult), "not sure when this wouldn't be true"

        # try taking an action, unless it fails to parse
        # TODO: make some kind of FailedAction so we can ask for a retry
        # TODO: early stop when too many action parsing errors occur
        try:
            action = env.parse_action_string(solver_output.output)
            logger.info(f"\nAction:\n{action}\n=====\n")
            env_output = env.step(action)
            logger.info(f"\nEnv output observation:\n{env_output.observation}\n=====\n")
            trajectory.append(TrajectoryStep(action, env_output))

        except ActionParsingError:
            logger.warning(f"\nFailed to parse action string:\n{solver_output.output}\n=====\n")
            action = ParsingErrorAction(
                is_stop=False,
                raw_prediction=solver_output.output,
                parsed_prediction="ERROR: Failed to parse action. Make sure to wrap the arguments inside [] and the whole action inside a pair of triple backticks (```).",
            )
            # NOTE: we don't take a step in the environment because the action was invalid
            trajectory.append(TrajectoryStep(action, env_output))

    return trajectory


def _task_state_from_trajectory(
    trajectory: Trajectory, goal: str, task_description_template=DEFAULT_TASK_DESCRIPTION_TEMPLATE
) -> MWTTaskState:
    action = trajectory[-1].action
    action_string = "None" if action is None else action.parsed_prediction
    env_output = trajectory[-1].env_output
    observation = env_output.observation.data
    messages = _messages_from_trajectory(trajectory)

    task_description = task_description_template.format(goal=goal)

    # TODO: clean this up somehow, so I don't have to check
    if isinstance(env_output, BrowserEnvOutput):
        return MWTTaskState(
            task_description=task_description,
            messages=messages,
            previous_action=action_string,
            observation=observation,
            url=env_output.info.page.url,
            goal=goal,
            current_state=None,  # todo: use this?
        )
    elif isinstance(env_output, BashEnvOutput):
        return MWTTaskState(
            task_description=task_description,
            messages=messages,
            previous_action=action_string,
            observation=observation,
            url=None,
            goal=goal,
            current_state=None,
        )
    else:
        # returns from BashBrowserEnv should be either BrowserEnvOutput
        # or BashEnvOutput, depending on which action was just performed
        raise ValueError(f"Unknown env output type {type(env_output)}")


def _messages_from_trajectory(trajectory: Trajectory) -> list[Message]:
    """Build a list of messages from the trajectory.
    We don't have to include the initial instructions (i.e. the task description)
    so we'll just make a list of observation (user messages) and action (assistant messages).
    If the action is None we skip it, since that means it was the initial observation step.
    """
    messages = []
    for step in trajectory:
        action = step.action
        observation = step.env_output.observation
        if action is not None:
            messages.append(Message(role="assistant", content=action.parsed_prediction))
        messages.append(Message(role="user", content=observation.data))
    return messages


def _episode_should_continue(trajectory: Trajectory, early_stop_config: EarlyStopConfig) -> bool:
    """
    Either the environment decides that the episode is over, or the agent
    issues a stop action.  The agent usually decides when the episode is over,
    unless it's caught in a loop of repeating actions.
    """

    last_step = trajectory[-1]
    env_should_continue = not last_step.env_output.done
    no_stop_action = last_step.action is None or not last_step.action.is_stop
    should_stop_early = should_early_stop(trajectory, early_stop_config)
    return (
        env_should_continue
        and no_stop_action  # environment hasn't emitted done
        and not should_stop_early  # agent hasn't emitted stop  # early stopping conditions aren't met
    )


def evaluate_trajectory(
    trajectory: Trajectory,
    env: LLMAgentEnv,
    experiment_config: ExperimentConfig,
) -> ExperimentResult:
    evaluator = evaluator_router(experiment_config)
    score = evaluator(
        trajectory=trajectory,
        env=env,
        experiment_config=experiment_config,
    )
    return ExperimentResult(
        score=score,
        trajectory=trajectory,
        env=env,
        experiment_config=experiment_config,
    )


def record_result(
    result: ExperimentResult,
    args: argparse.Namespace,
) -> None:
    """TODO: add more features to this, such as creating a render
    like the original WebArena does"""
    trajectory_path = Path(args.result_dir) / "trajectory.txt"
    with trajectory_path.open("w") as f:
        f.write(result.trajectory.pretty_string())


def setup_env(
    experiment_config: ExperimentConfig,
    session: Session,
) -> LLMAgentEnv:
    """TODO: move this and constituent functions to separate file/dir"""
    # TODO: change to match-case statement in Python 3.10
    if isinstance(experiment_config, BashBrowserExperimentConfig):
        env = setup_bash_browser_env(experiment_config, session)
    elif isinstance(experiment_config, BrowserExperimentConfig):
        env = setup_browser_env(experiment_config, session)
    elif isinstance(experiment_config, BashExperimentConfig):
        env = setup_bash_env(experiment_config, session)
    else:
        raise ValueError(f"Unknown env type {type(experiment_config)}")
    return env


def setup_browser_env(
    experiment_config: BrowserExperimentConfig,
    session,
) -> BrowserEnv:
    env = BrowserEnv(
        session=session,
        headless=experiment_config.headless,
        slow_mo=experiment_config.slow_mo,
        observation_type=experiment_config.observation_type,
        current_viewport_only=experiment_config.current_viewport_only,
        viewport_size={
            "width": experiment_config.viewport_width,
            "height": experiment_config.viewport_height,
        },
        save_trace_enabled=experiment_config.save_trace_enabled,
        sleep_after_execution=experiment_config.sleep_after_execution,
    )
    return env


def setup_bash_env(
    experiment_config: BashExperimentConfig,
    session: Session,
) -> BashEnv:
    env = BashEnv(session)
    return env


def setup_bash_browser_env(
    experiment_config: BashBrowserExperimentConfig,
    session,
) -> BashBrowserEnv:
    env = BashBrowserEnv(
        session=session,
        # for browser env
        headless=experiment_config.headless,
        slow_mo=experiment_config.slow_mo,
        observation_type=experiment_config.observation_type,
        current_viewport_only=experiment_config.current_viewport_only,
        viewport_size={
            "width": experiment_config.viewport_width,
            "height": experiment_config.viewport_height,
        },
        save_trace_enabled=experiment_config.save_trace_enabled,
        sleep_after_execution=experiment_config.sleep_after_execution,
    )
    return env


def should_early_stop(trajectory: Trajectory, es_config: EarlyStopConfig) -> bool:
    """Check whether we should stop early"""
    if len(trajectory) >= es_config.max_steps:
        return True

    # TODO: implement parsing failure early stopping
    # if _check_repeated_parsing_failure(trajectory, es_config.parsing_failure):
    # return True

    if _check_repeated_equivalent_actions(trajectory, es_config.repeating_action):
        return True

    # if no conditions met, don't early stop
    return False


def _check_repeated_equivalent_actions(trajectory: Trajectory, repeating_action: int) -> bool:
    recent_steps = trajectory[-repeating_action:]
    # if the len is different, then we haven't had enough actions for this condition yet
    # (also have to check for None action at the start)
    if len(recent_steps) == repeating_action and recent_steps[0].action is not None:
        reference_action = recent_steps[0].action
        if all(is_equivalent(step.action, reference_action) for step in recent_steps):  # type: ignore (it can't be none)
            return True
    return False
