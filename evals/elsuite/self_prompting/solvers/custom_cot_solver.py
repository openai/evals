from typing import Any, Dict, Optional

from evals.completion_fns.openai import OpenAIChatCompletionFn
from evals.solvers.solver import OpenAISolver, SolverResult
from evals.task_state import TaskState

DEFAULT_COT_PRE_TEMPLATE = "{instructions}"
DEFAULT_COT_POST_TEMPLATE = "Before answering, reason in a step-by-step manner as to get the right answer, then conclude with the answer."
DEFAULT_EXTRACT_ANSWER_TEMPLATE = (
    "Given the above reasoning, the answer in the format requested by the question is:"
)


class CustomCoTSolver(OpenAISolver):
    def __init__(
        self,
        cot_options: Dict[str, Any] = {},
        cot_pre_template: str = DEFAULT_COT_PRE_TEMPLATE,
        cot_post_template: str = DEFAULT_COT_POST_TEMPLATE,
        extract_options: Dict[str, Any] = {},
        extract_template: str = DEFAULT_EXTRACT_ANSWER_TEMPLATE,
        valid_answers: Optional[list[str]] = None,
        **kwargs,
    ):
        super().__init__(
            completion_fn_options=extract_options,
            valid_answers=valid_answers,
        )

        self.cot_completion_fn = OpenAIChatCompletionFn(
            **cot_options,
        )
        self.cot_pre_template = cot_pre_template
        self.cot_post_template = cot_post_template

        self.extract_completion_fn = OpenAIChatCompletionFn(**self.completion_fn_options)
        self.extract_template = extract_template

    def _solve(
        self,
        task_state: TaskState,
        **kwargs,
    ) -> SolverResult:

        # Reasoning step
        cot_pre = self.cot_pre_template.format(instructions=task_state.task_description)
        cot_post = self.cot_post_template
        msgs = []
        if cot_pre != "":
            msgs.append({"role": "system", "content": cot_pre})
        msgs += [msg.to_dict() for msg in task_state.messages]
        if cot_post != "":
            msgs.append({"role": "system", "content": cot_post})
        reasoning_output = self.cot_completion_fn(prompt=msgs, **kwargs).get_completions()[0]

        # Extract answer step
        msgs = msgs + [
            {"role": "assistant", "content": reasoning_output},
            {"role": "assistant", "content": self.extract_template},
        ]
        extracted_answer = self.extract_completion_fn(prompt=msgs, **kwargs).get_completions()[0]

        return SolverResult(
            output=extracted_answer,
            reasoning_output=reasoning_output,
        )

    @property
    def name(self) -> str:
        return f"SelfPromptingCoT_{self.cot_completion_fn.model}_{self.extract_completion_fn.model}"
