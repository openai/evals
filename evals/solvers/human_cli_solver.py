from copy import deepcopy
from typing import Any

from evals.record import record_event, record_sampling
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
        explain: bool = False,
        postprocessors: list[str] = [],
        registry: Any = None,
    ):
        """
        Args:
            input_prompt: Prompt to be printed before the user input.
                If None, no prompt is printed.
            explain: If True, print structured context and output details for
                human-in-the-loop debugging.
        """
        super().__init__(postprocessors=postprocessors)
        self.input_prompt = input_prompt
        self.explain = explain

    def _get_messages(self, task_state: TaskState) -> list[Message]:
        return [Message("system", task_state.task_description)] + task_state.messages

    def _get_prompt(self, task_state: TaskState) -> str:
        msgs = self._get_messages(task_state)
        return "\n".join([f"{msg.role}: {msg.content}" for msg in msgs]) + f"\n{self.input_prompt}"

    def _print_explain_context(self, task_state: TaskState, prompt: str) -> None:
        print("================ TASK CONTEXT (system) ================")
        print(task_state.task_description)
        print()
        print("================ MESSAGE HISTORY ================")
        if task_state.messages:
            for idx, msg in enumerate(task_state.messages):
                print(f"[{idx}] {msg.role}: {msg.content}")
        else:
            print("(no prior messages)")
        print()
        print("================ FINAL PROMPT STRING ================")
        print(prompt)
        print()
        print("================ AWAITING HUMAN INPUT ================")

    def _print_explain_outcome(self, prompt: str, raw_answer: str, final_answer: str) -> None:
        print()
        print("================ SAMPLING RECORD ================")
        print("model: human")
        print(f"prompt_chars: {len(prompt)}")
        print(f"answer_chars: {len(raw_answer)}")
        print()
        print("================ OUTPUT ================")
        print(f"raw_answer: {raw_answer}")
        print(f"final_answer: {final_answer}")

    def _solve(self, task_state: TaskState, **kwargs) -> SolverResult:
        prompt = self._get_prompt(task_state)
        if self.explain:
            self._print_explain_context(task_state, prompt)

        cli_prompt = self.input_prompt if self.explain else prompt
        answer = input(cli_prompt)

        record_sampling(
            prompt=prompt,
            sampled=answer,
            model="human",
        )

        return SolverResult(answer, prompt=prompt)

    def __call__(self, task_state: TaskState, **kwargs) -> SolverResult:
        res = self._solve(deepcopy(task_state), **kwargs)
        raw_output = res.output

        if hasattr(self, "postprocessors"):
            for postprocessor in self.postprocessors:
                prev_output = res.output
                res = postprocessor(res)
                record_event(
                    "postprocessor",
                    {
                        "name": postprocessor.__class__.__name__,
                        "input": prev_output,
                        "output": res.output,
                    },
                )

        if self.explain:
            self._print_explain_outcome(res.metadata["prompt"], raw_output, res.output)

        return res

    @property
    def name(self) -> str:
        return "human"
