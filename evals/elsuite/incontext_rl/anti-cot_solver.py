from typing import Any
from evals.solvers.solver import NestedSolver, Solver, SolverResult, SolverSpec
from evals.task_state import Message, TaskState

ANTI_COT_TEMPLATE = "RESPOND ONLY WITH YOUR FINAL ANSWER IN THE FORMAT REQUESTED. DO NOT OUTPUT ANY ADDITIONAL REASONING OR TEXT."

class AntiCoTSolver(NestedSolver):
    """
    Instructs the model to not do any further reasoning and just respond with the final answer.
    """

    def __init__(
        self,
        solver: SolverSpec,
        registry: Any = None,
    ):
        super().__init__(solver=solver)

    @property
    def solver(self) -> Solver:
        return self.get_solver("solver")

    def _solve(
        self,
        task_state: TaskState,
        **kwargs,
    ) -> SolverResult:
        task_state.messages += (
            [
                Message(role="system", content=ANTI_COT_TEMPLATE),
            ]
        )
        solver_result = self.solver(task_state=task_state, **kwargs)
        return solver_result

    @property
    def name(self) -> str:
        return f"Anti-CoT_{self.solver.name}"
