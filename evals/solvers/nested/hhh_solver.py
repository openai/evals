from typing import Any

from evals.solvers.prompts.hhh import HHH_MSGS
from evals.solvers.solver import NestedSolver, Solver, SolverResult, SolverSpec
from evals.task_state import Message, TaskState


class HHHSolver(NestedSolver):
    """
    Adds Helpful, Harmless and Honest (HHH) messages (Bai et al., 2022) to the
    prompt history. This is especially useful for completion models that are
    not instruction- or chat-tuned, as the context encourages the model to
    generate a response that is consistent with a HHH chatbot assistant.
    """

    def __init__(
        self,
        solver: SolverSpec,
        postprocessors: list[str] = [],
        registry: Any = None,
    ):
        super().__init__(postprocessors=postprocessors, solver=solver)

    @property
    def solver(self) -> Solver:
        return self.get_solver("solver")

    def _solve(
        self,
        task_state: TaskState,
        **kwargs,
    ) -> SolverResult:
        task_state.messages = (
            HHH_MSGS[1:]  # The first HHH message will go into the task_description
            + [
                Message(role="system", content=task_state.task_description),
            ]
            + task_state.messages
        )
        task_state.task_description = HHH_MSGS[0].content  # Below are a series of dialogues...

        solver_result = self.solver(task_state=task_state, **kwargs)
        return solver_result

    @property
    def name(self) -> str:
        return f"{self.solver.name}_hhh"
