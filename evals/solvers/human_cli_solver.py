from typing import Any

from evals.record import record_sampling
from evals.solvers.solver import Solver, SolverResult
from evals.task_state import Message, TaskState


class HumanCliSolver(Solver):
    """Solver that prints prompts to the command line and reads input from it.

    NOTE: With more than a single thread messages from different threads will mix,
          so this makes sense only with EVALS_SEQUENTIAL=1.
    """

    def __init__(
        self,
        input_prompt: str = "assistant (you): ",
        postprocessors: list[str] = [],
        registry: Any = None,
    ):
        """
        Args:
            input_prompt: Prompt to be printed before the user input.
                If None, no prompt is printed.
        """
        super().__init__(postprocessors=postprocessors)
        self.input_prompt = input_prompt

    def _solve(self, task_state: TaskState, **kwargs) -> SolverResult:
        msgs = [Message("system", task_state.task_description)]
        msgs += task_state.messages

        prompt = (
            "\n".join([f"{msg.role}: {msg.content}" for msg in msgs]) + f"\n{self.input_prompt}"
        )
        answer = input(prompt)

        record_sampling(
            prompt=prompt,
            sampled=answer,
            model="human",
        )

        return SolverResult(answer)

    @property
    def name(self) -> str:
        return "human"
