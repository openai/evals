from typing import Any, Dict, Union

from evals.completion_fns.openai import OpenAICompletionFn
from evals.solvers.prompts.cot import DEFAULT_COT_TEMPLATE, DEFAULT_EXTRACT_ANSWER_TEMPLATE
from evals.solvers.prompts.hhh import HHH_PROMPT, render_messages
from evals.solvers.solver import OpenAISolver, SolverResult
from evals.task_state import TaskState


class OpenAICompletionHHHCoTSolver(OpenAISolver):
    def __init__(
        self,
        cot_options: Dict[str, Any] = {},
        cot_template: str = DEFAULT_COT_TEMPLATE,
        extract_options: Dict[str, Any] = {},
        extract_template: str = DEFAULT_EXTRACT_ANSWER_TEMPLATE,
        fixed_start: str = "",
        valid_answers: Union[list[str], None] = None,
        **kwargs,
    ):
        super().__init__(
            completion_fn_options=extract_options,
            valid_answers=valid_answers,
        )

        self.cot_completion_fn = OpenAICompletionFn(**cot_options)
        self.cot_template = cot_template

        self.extract_completion_fn = OpenAICompletionFn(**self.completion_fn_options)
        self.extract_template = extract_template

        self.fixed_start = fixed_start

    def __call__(
        self,
        task_state: TaskState,
        **kwargs,
    ) -> SolverResult:

        # Reasoning step
        prompt = (
            HHH_PROMPT
            + f"""System:  {task_state.task_description}

{render_messages(task_state.messages)}System:  {self.cot_template}

Assistant:  """
        )
        reasoning_output_raw = self.cot_completion_fn(prompt=prompt, **kwargs).get_completions()[0]
        # Parse response up to the next "Human:", "System:", or "-----"
        reasoning_output = (
            reasoning_output_raw.split("Human:")[0].split("System:")[0].split("-----")[0].strip()
        )

        # Extract answer step
        prompt = (
            HHH_PROMPT
            + f"""System:  {task_state.task_description}

{render_messages(task_state.messages)}System:  {self.cot_template}

Assistant:  {reasoning_output}

System:  {self.extract_template}

Assistant:  {self.fixed_start}"""
        )
        answer_output_raw = self.extract_completion_fn.__call__(
            prompt=prompt, **kwargs
        ).get_completions()[0]
        # Parse response up to the next "Human:", "System:", or "-----"
        answer_output = (
            answer_output_raw.split("Human:")[0].split("System:")[0].split("-----")[0].strip()
        )
        answer_output = self.fixed_start + answer_output

        return SolverResult(
            output=answer_output,
            reasoning_output=reasoning_output,
            answer_output_raw=answer_output_raw,
        )

    @property
    def name(self) -> str:
        return f"CoT_{self.cot_completion_fn.model}_{self.extract_completion_fn.model}"
