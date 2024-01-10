from dataclasses import dataclass
from typing import Any, Dict, List, Union

from evals.completion_fns.openai import OpenAIChatCompletionFn
from evals.solvers.prompts.cot import DEFAULT_COT_TEMPLATE, DEFAULT_EXTRACT_ANSWER_TEMPLATE
from evals.solvers.solver import OpenAISolver, SolverResult
from evals.task_state import Message, TaskState


@dataclass
class Interaction:
    #   All messages we've seen (except for the task_description)
    messages: List[Message]

    #   IDs of the CoT private internal messages
    private_messages_ids: List[int]


class OpenAIChatCompletionCoTSolver(OpenAISolver):
    def __init__(
        self,
        cot_options: Dict[str, Any] = {},
        cot_template: str = DEFAULT_COT_TEMPLATE,
        extract_options: Dict[str, Any] = {},
        extract_template: str = DEFAULT_EXTRACT_ANSWER_TEMPLATE,
        valid_answers: Union[list[str], None] = None,
        persistent_memory: bool = True,
        private_interaction_length: int = 3,
        **kwargs,
    ):
        super().__init__(
            completion_fn_options=extract_options,
            valid_answers=valid_answers,
        )

        self.cot_completion_fn = OpenAIChatCompletionFn(
            **cot_options,
        )
        self.cot_template = cot_template

        self.extract_completion_fn = OpenAIChatCompletionFn(**self.completion_fn_options)
        self.extract_template = extract_template

        self.persistent_memory = persistent_memory
        self.last_interaction = None
        self.private_interaction_length = private_interaction_length

    def __call__(
        self,
        task_state: TaskState,
        **kwargs,
    ) -> SolverResult:

        past_messages = (
            task_state.messages
            if not self.persistent_memory
            else self._persistent_memory_past_messages(task_state)
        )

        # Reasoning step
        msgs = (
            [
                {"role": "system", "content": task_state.task_description},
            ]
            + [msg.to_dict() for msg in past_messages]
            + [
                {"role": "system", "content": self.cot_template},
            ]
        )
        reasoning_output = self.cot_completion_fn(prompt=msgs, **kwargs).get_completions()[0]

        # Extract answer step
        msgs = msgs + [
            {"role": "assistant", "content": reasoning_output},
            {"role": "assistant", "content": self.extract_template},
        ]
        extracted_answer = self.extract_completion_fn(prompt=msgs, **kwargs).get_completions()[0]

        #   Save the interaction
        interaction_messages = [Message(**msg) for msg in msgs[1:]] + [
            Message("assistant", extracted_answer)
        ]
        num_interaction_messages = len(interaction_messages)
        private_messages_ids = (
            [] if self.last_interaction is None else self.last_interaction.private_messages_ids
        )
        private_messages_ids += list(
            range(
                num_interaction_messages - self.private_interaction_length - 1,
                num_interaction_messages - 1,
            )
        )
        self.last_interaction = Interaction(interaction_messages, private_messages_ids)

        return SolverResult(
            output=extracted_answer,
            reasoning_output=reasoning_output,
        )

    @property
    def name(self) -> str:
        return f"CoT_{self.cot_completion_fn.model}_{self.extract_completion_fn.model}"

    def _persistent_memory_past_messages(self, task_state: TaskState) -> List[Message]:
        if self.last_interaction is None:
            return task_state.messages

        #   Check if task_state matches our last interaction
        interaction = self.last_interaction
        task_state_message_ix = 0
        for our_message_ix in range(0, len(interaction.messages)):
            if our_message_ix in interaction.private_messages_ids:
                continue
            else:
                if (
                    task_state.messages[task_state_message_ix]
                    != interaction.messages[our_message_ix]
                ):
                    raise ValueError(
                        (
                            f"task_state message {task_state_message_ix} different than the corresponding message "
                            "in the interaction history. "
                            "Such scenario is not supported by a CoT solver with peristent_memory = True"
                        )
                    )
                task_state_message_ix += 1

        #   Everything's fine!
        return interaction.messages + task_state.messages[task_state_message_ix:]
