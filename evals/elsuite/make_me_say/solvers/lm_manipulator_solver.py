from importlib import import_module

from evals.elsuite.make_me_say.solvers.prompts import (
    PERSUADER_MSG,
    COT_MESSAGE,
    EXTRACT_MESSAGE,
    TIME_REMINDER_MSG,
)
from evals.solvers.nested.cot_solver import CoTSolver
from evals.solvers.solver import Solver, SolverResult
from evals.task_state import Message, TaskState


class LMManipulatorSolver(Solver):
    def __init__(
        self,
        base_solver_class: str,
        base_solver_args: dict,
        lm_system_prompt: bool = False,
        time_reminder: bool = False,
        **kwargs,
    ):
        module_name, class_name = base_solver_class.split(":")
        module = import_module(module_name)
        cls = getattr(module, class_name)

        self.lm_system_prompt = lm_system_prompt
        self.time_reminder = time_reminder

        # Add more args to base solver if we're using CoT
        base_solver_args_extra = {}
        if issubclass(cls, CoTSolver):
            base_solver_args_extra = {
                "cot_template": COT_MESSAGE,
                "extract_template": EXTRACT_MESSAGE,
            }

            # 3 by default due to CoT (CoT prompt, model response, "now give answer")
            private_interaction_length = 3
            if time_reminder:
                private_interaction_length += 1

            base_solver_args_extra["private_interaction_length"] = private_interaction_length

        self.base_solver = cls(
            **base_solver_args,
            **base_solver_args_extra,
        )

    def name(self):
        return "Scaffolded-LM-Solver"

    def _solve(self, task_state: TaskState, **kwargs) -> SolverResult:
        # Optional additional message for better LM capabilities. Only append if
        # this is start of conversaiton, otherwise this is included in memory
        if self.lm_system_prompt:
            max_turns_or_time = task_state.current_state["max_turns_or_time"]
            keyword = task_state.current_state["keyword"]
            content = PERSUADER_MSG.format(duration=max_turns_or_time, keyword=keyword)
            task_state.task_description += f"\n{content}"

        # Add time reminder system message
        if self.time_reminder:
            turns_or_time_left = task_state.current_state["turns_or_time_left"]
            if isinstance(turns_or_time_left, int):
                msg = TIME_REMINDER_MSG.format(turns_or_time=f"{turns_or_time_left} turn(s)")
            elif isinstance(turns_or_time_left, float):
                if turns_or_time_left < 2:
                    seconds_left = int(60 * turns_or_time_left)
                    msg = TIME_REMINDER_MSG.format(turns_or_time=f"{seconds_left} second(s)")
                else:
                    msg = TIME_REMINDER_MSG.format(
                        turns_or_time=f"{round(turns_or_time_left, 1)} minute(s)"
                    )
            else:
                assert (
                    False
                ), "turns_or_time_left must be of time int if indicating turns, else float if storing time"

            msg = Message(role="system", content=msg)
            task_state.messages.append(msg)

        return self.base_solver(task_state, **kwargs)
