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
        explain: bool = False,
    ):
        """
        Args:
            input_prompt: Prompt to be printed before the user input.
                If None, no prompt is printed.
            explain: Print structured prompt, sampling, and output details for
                human-baseline debugging and audit workflows.
        """
        super().__init__(postprocessors=postprocessors)
        self.input_prompt = input_prompt
        self.explain = explain

    def __call__(self, task_state: TaskState, **kwargs) -> SolverResult:
        result = super().__call__(task_state, **kwargs)
        if self.explain:
            print("\n================ FINAL OUTPUT ================")
            print(result.output)
        return result

    def _solve(self, task_state: TaskState, **kwargs) -> SolverResult:
        msgs = [Message("system", task_state.task_description)]
        msgs += task_state.messages

        prompt = (
            "\n".join([f"{msg.role}: {msg.content}" for msg in msgs]) + f"\n{self.input_prompt}"
        )

        if self.explain:
            self._print_prompt_explanation(task_state, prompt)
            answer = input(self.input_prompt)
        else:
            answer = input(prompt)

        record_sampling(
            prompt=prompt,
            sampled=answer,
            model="human",
        )

        if self.explain:
            print("\n================ SAMPLING RECORD ================")
            print("model: human")
            print(f"prompt_chars: {len(prompt)}")
            print(f"answer_chars: {len(answer)}")
            print("\n================ RAW OUTPUT ================")
            print(answer)

        return SolverResult(answer)

    @staticmethod
    def _print_prompt_explanation(task_state: TaskState, prompt: str) -> None:
        print("================ TASK CONTEXT (system) ================")
        print(task_state.task_description)
        print("\n================ MESSAGE HISTORY ================")
        if task_state.messages:
            for i, msg in enumerate(task_state.messages):
                print(f"[{i}] {msg.role}: {msg.content}")
        else:
            print("(empty)")
        print("\n================ FINAL PROMPT STRING (exact) ================")
        print(prompt)
        print("\n================ AWAITING HUMAN INPUT ================")

    @property
    def name(self) -> str:
        return "human"
