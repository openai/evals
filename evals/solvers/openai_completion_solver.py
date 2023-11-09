from typing import Any, Dict, Union

from evals.completion_fns.openai import OpenAICompletionFn
from evals.solvers.solver import OpenAISolver, SolverResult
from evals.task_state import TaskState


class OpenAICompletionSolver(OpenAISolver):
    def __init__(
        self,
        completion_fn_options: Dict[str, Any] = {},
        valid_answers: Union[list[str], None] = None,
        **kwargs,
    ):
        super().__init__(
            completion_fn_options=completion_fn_options,
            valid_answers=valid_answers,
        )

        self.completion_fn = OpenAICompletionFn(
            **self.completion_fn_options,
        )

    def __call__(
        self,
        task_state: TaskState,
        **kwargs,
    ) -> SolverResult:

        msgs = [
            {"role": "system", "content": task_state.task_description},
        ] + [msg.to_dict() for msg in task_state.messages]

        completion_result = self.completion_fn(prompt=msgs, **kwargs)
        return SolverResult(completion_result.get_completions()[0])

    @property
    def name(self) -> str:
        return self.completion_fn.model
