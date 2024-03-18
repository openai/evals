import json
import time
from dataclasses import dataclass, replace
from logging import getLogger
from pathlib import Path

from evals.elsuite.ml_agent_bench.actions import get_action, is_valid_action
from evals.elsuite.ml_agent_bench.auto_marking import EvaluationResult, grade_submission
from evals.elsuite.ml_agent_bench.environment import Environment
from evals.elsuite.ml_agent_bench.prompts import get_task_description
from evals.elsuite.ml_agent_bench.schema import ActionInfo
from evals.solvers.solver import Solver
from evals.task_state import Message, TaskState

logger = getLogger(__name__)


@dataclass(frozen=True)
class Step:
    step_idx: int
    action: dict[str, str]
    observation: str


@dataclass(frozen=True)
class TaskStateMetadata:
    history_steps: tuple[Step, ...]
    actions: dict[str, ActionInfo]
    max_steps_in_context: int
    max_retries: int
    max_steps: int
    log_dir: Path
    env: Environment


@dataclass(frozen=True)
class FunctionCall:
    name: str
    args: dict[str, str]


def run(
    solver: Solver,
    task_name: str,
    research_problem: str,
    log_dir: Path,
    work_dir: Path,
    max_steps: int,
    max_time: int,
    max_seconds_per_step: int,
    device: int = 0,
    python_command: str = "python",
    resume: bool = False,
    resume_step: int = 0,
    max_steps_in_context: int = 3,
    max_retries: int = 5,
) -> EvaluationResult:
    """Evaluates the solver on a given task."""

    env = Environment(
        log_dir=log_dir / "env_log",
        work_dir=work_dir / task_name,
        task=task_name,
        python_command=python_command,
        resume=resume,
        resume_step=resume_step,
        device=device,
        max_steps=max_steps,
        max_time=max_time,
        solver=solver,
    )

    task_description = get_task_description(research_problem)

    logger.info(task_description)

    messages = [
        Message(
            role="system",
            content=f"You have a maximum of {max_steps} steps to solve the task. "
            f"Each step is subject to a maximum time limit of {max_seconds_per_step} "
            f"seconds. Additionally, your entire attempt is subject to a maximum "
            f"time limit of {max_time} seconds.",
        ),
    ]

    task_state = TaskState(
        task_description=task_description,
        messages=messages,
        current_state=TaskStateMetadata(
            history_steps=(),
            actions=env.action_infos,
            max_steps_in_context=max_steps_in_context,
            max_retries=max_retries,
            max_steps=max_steps,
            log_dir=log_dir,
            env=env,
        ),
    )

    start_time = time.time()

    for step in range(max_steps):
        time_elapsed = time.time() - start_time
        time_remaining = max_time - time_elapsed

        task_state = replace(
            task_state,
            messages=task_state.messages
            + [
                Message(
                    role="system",
                    content=f"You have {time_remaining:.2f} seconds and {max_steps - step} steps remaining.",
                ),
            ],
        )

        result = solver(task_state)
        new_messages = [Message(role="assistant", content=result.output)]

        action = get_action(result.output)

        if action is None or not is_valid_action(action):
            logger.info(
                f"Step {step}: [invalid response; no action executed]\n\nAssistant:\n{result.output}"
            )

            new_messages = task_state.messages + [
                Message(
                    role="assistant",
                    content=result.output,
                ),
                Message(
                    role="system",
                    content="No valid action found! The list of valid actions was specified at the start; please pick an action from that list.",
                ),
            ]

            task_state = TaskState(
                task_description=task_state.task_description,
                messages=new_messages,
                current_state=task_state.current_state,
            )

            continue

        logger.info(f"\n\nAction: {action.name}\nAction Input: {json.dumps(action.args)}\n")

        new_max_seconds_per_step = min(max_seconds_per_step, time_remaining)

        if attempted_to_use_stable_baselines(result.output):
            observation = "The action has been terminated because it used the `stable_baselines` "
            "library, which is not allowed. Please write your implementation from scratch."
        elif time_remaining > 0:
            observation = env.execute(action, max_seconds_per_step=new_max_seconds_per_step)
        else:
            observation = "Time's up! Your attempt has now exceeded the maximum time limit "
            f"of {max_time} seconds. The last action attempted was not executed, "
            "and your current solution will be graded."

        new_messages = task_state.messages + [
            Message(
                role="assistant",
                content=result.output,
            ),
            Message(
                role="system",
                content=f"Observation:\n\n```\n{observation}\n```",
            ),
        ]

        new_history_steps = task_state.current_state.history_steps + (
            {
                "step_idx": step,
                "action": {
                    "Action": action.name,
                    "Action Input": json.dumps(action.args, indent=4),
                },
                "observation": observation,
            },
        )

        new_task_state_metadata = replace(
            task_state.current_state,
            history_steps=new_history_steps,
        )

        task_state = TaskState(
            task_description=task_state.task_description,
            messages=new_messages,
            current_state=new_task_state_metadata,
        )

        logger.info(f"\n\nObservation:\n```\n{observation}\n```\n")

        env.save(step)

        if env.is_done():
            break

    env.save("final")

    result = grade_submission(log_dir=log_dir, task_name=task_name)

    return result


def attempted_to_use_stable_baselines(s: str) -> bool:
    s = s.lower()  # be case-insensitive

    if "stable" in s and "baseline" in s:
        return True

    return False
