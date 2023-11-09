from typing import Any, Dict, Union

from evals.completion_fns.openai import OpenAICompletionFn
from evals.solvers.prompts.hhh import HHH_PROMPT, render_messages
from evals.solvers.solver import OpenAISolver, SolverResult
from evals.task_state import TaskState


class OpenAICompletionHHHSolver(OpenAISolver):
    def __init__(
        self,
        completion_fn_options: Dict[str, Any] = {},
        valid_answers: Union[list[str], None] = None,
        fixed_start: str = "",
        **kwargs,
    ):
        super().__init__(
            completion_fn_options=completion_fn_options,
            valid_answers=valid_answers,
        )

        self.completion_fn = OpenAICompletionFn(
            **self.completion_fn_options,
        )

        self.fixed_start = fixed_start

    def __call__(
        self,
        task_state: TaskState,
        **kwargs,
    ) -> SolverResult:

        prompt = f"{HHH_PROMPT}System:  {task_state.task_description}\n\n{render_messages(task_state.messages)}Assistant:  "
        prompt = prompt + self.fixed_start

        completion_result = self.completion_fn(prompt=prompt, **kwargs)
        raw_response = completion_result.get_completions()[0]

        # Parse response up to the next "Human:", "System:", or "-----"
        parsed_response = (
            raw_response.split("Human:")[0].split("System:")[0].split("-----")[0].strip()
        )
        parsed_response = self.fixed_start + parsed_response

        return SolverResult(parsed_response)

    @property
    def name(self) -> str:
        return self.completion_fn.model
