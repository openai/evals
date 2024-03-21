import random
from typing import Any

from evals.elsuite.track_the_stat import utils
from evals.solvers.solver import NestedSolver, Solver, SolverResult, SolverSpec
from evals.task_state import Message, TaskState


class ExplicitStateSolver(NestedSolver):
    def __init__(
        self,
        underlying_solver: SolverSpec,
        state_role: str = "assistant",
        *args,
        **kwargs,
    ):
        super().__init__(underlying_solver=underlying_solver, *args, **kwargs)
        self.state_role = state_role

    @property
    def underlying_solver(self) -> Solver:
        return self.get_solver("underlying_solver")

    def _render_state(self, current_state: dict) -> str:
        rendered_state_string = f"{current_state['state_label']}\n{current_state['state_data']}"
        return rendered_state_string

    def _build_message(self, task_state: TaskState) -> str:
        message_string = "The current state, useful for solving the task\n" + self._render_state(
            task_state.current_state
        )
        return Message(role=self.state_role, content=message_string)

    def _solve(self, task_state: TaskState) -> SolverResult:
        precomputed_state_message = self._build_message(task_state)
        task_state.messages.append(precomputed_state_message)

        solver_result = self.underlying_solver(task_state=task_state)
        return solver_result


class RandomBaselineSolver(Solver):
    def __init__(self, registry: Any = None, *args, **kwargs):
        super().__init__()

    def _solve(self, task_state: TaskState) -> SolverResult:
        task = task_state.current_state["task_name"]
        random_output = self._task_solve(task, task_state)
        solver_result = SolverResult(output=f"[{task}: {random_output}]")
        return solver_result

    def _task_solve(self, task: str, task_state: TaskState) -> str:
        if task == "mode":
            return self._mode_solve(task_state)
        elif task == "median":
            return self._median_solve(task_state)

    def _mode_solve(self, task_state: TaskState) -> str:
        """
        Picks a random number from the numbers seen so far
        """
        numbers = list(task_state.current_state["state_data"].keys())
        random_mode = random.choice(numbers)
        return str(random_mode)

    def _median_solve(self, task_state: TaskState) -> str:
        """
        Picks a random number from the numbers seen so far
        (in case of even number of numbers, picks the average of two random numbers)
        """
        numbers = task_state.current_state["state_data"]
        if len(numbers) % 2 == 0:
            random_1, random_2 = random.choices(numbers, k=2)
            random_median = (random_1 + random_2) / 2
        else:
            random_median = random.choice(numbers)
        return str(round(random_median, 1))


class TrackTheStatHuman(NestedSolver):
    def __init__(self, human_cli_solver: SolverSpec, *args, **kwargs):
        super().__init__(human_cli_solver=human_cli_solver, *args, **kwargs)

    @property
    def human_cli_solver(self) -> Solver:
        return self.get_solver("human_cli_solver")

    def _solve(self, task_state: TaskState) -> SolverResult:
        human_result = self.human_cli_solver(task_state=task_state)
        task = task_state.current_state["task_name"]
        # wrap the result in [<task>: <solver_result>] if not already wrapped
        output = utils.parse_solver_output(human_result.output, task)
        if output is None:  # there is a violation -- output is not wrapped
            return SolverResult(
                output=f"[{task}: {human_result.output}]",
            )
        else:  # no violation -- output is already wrapped
            return human_result
