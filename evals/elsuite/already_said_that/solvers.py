import random
from typing import Any

from evals.solvers.solver import NestedSolver, Solver, SolverResult, SolverSpec
from evals.task_state import TaskState


class RandomBaselineSolver(Solver):
    def __init__(self, registry: Any = None):
        super().__init__()

    def _solve(self, task_state: TaskState, **kwargs) -> SolverResult:
        answer = random.choice(["yes", "no"])
        return SolverResult(output=f"[answer: {answer}]")


class AlreadySaidThatHuman(NestedSolver):
    def __init__(self, human_cli_solver: SolverSpec, *args, **kwargs):
        super().__init__(human_cli_solver=human_cli_solver, *args, **kwargs)

    @property
    def human_cli_solver(self) -> Solver:
        return self.get_solver("human_cli_solver")

    def _solve(self, task_state: TaskState) -> SolverResult:
        human_result = self.human_cli_solver(task_state=task_state)
        answer = self._map_to_yesno(human_result.output)
        return SolverResult(
            output=f"[answer: {answer}]",
        )

    def _map_to_yesno(self, yesno_ish):
        """
        Maps Y, y, Yes,1, yes, N, n, No, no, 0 to yes or no, respectively.
        """
        if yesno_ish.lower() in {"y", "yes", "1"}:
            return "yes"
        elif yesno_ish.lower() in {"n", "no", "0"}:
            return "no"
        else:
            # for other answers, return the original answer
            return yesno_ish
