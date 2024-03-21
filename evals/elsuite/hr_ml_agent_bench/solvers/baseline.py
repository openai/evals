"""This is a simple agent inspired by MLAgentBench's `SimpleActionAgent`"""

from functools import cache
from logging import getLogger

import tiktoken

from evals.registry import Registry, n_ctx_from_model_name
from evals.solvers.openai_solver import OpenAISolver
from evals.solvers.solver import Solver, SolverResult
from evals.task_state import Message, TaskState

logger = getLogger(__name__)


class SimpleActionAgent(Solver):
    def __init__(self, registry: Registry, completion_fn_kwargs: dict):
        self.completion_fn = OpenAISolver(**completion_fn_kwargs)
        self.ctx_len = n_ctx_from_model_name(self.completion_fn.model)

        assert (
            self.ctx_len is not None
        ), f"Expected the `ctx_len` to be set for model `{self.completion_fn.model}`."

    def _solve(self, task_state: TaskState, **kwargs) -> SolverResult:
        prompt = [
            Message(role="system", content=task_state.task_description),
            Message(
                role="system",
                content="Before deciding which action to take, you should "
                "reason aloud about what you are currently doing, what "
                "actions to perform and why.",
            ),
            Message(
                role="system",
                content="You should also check if GPUs are available and "
                "try your best to use them, if appropriate.",
            ),
            Message(role="system", content="Now let's start!"),
            *task_state.messages,
        ]

        step = len(task_state.current_state.history_steps)
        n_messages_to_include = task_state.current_state.max_steps_in_context

        for idx in range(max(0, step - n_messages_to_include), step):
            action_and_input = "".join(
                [
                    k + ": " + task_state.current_state.history_steps[idx]["action"][k]
                    for k in ["Action", "Action Input"]
                ]
            )

            reminder = f"This is turn number {idx+1} of {task_state.current_state.max_steps}. Remember to keep improving your solution until the turn limit is up, don't finish early!"

            observation = task_state.current_state.history_steps[idx]["observation"]

            encoder = self.get_encoder()
            max_tokens_in_observation = min(self.ctx_len // 8, 2**12)
            n_tokens_in_observation = len(encoder.encode(observation))

            if n_tokens_in_observation >= max_tokens_in_observation:
                logger.info(
                    f"Truncating observation. {max_tokens_in_observation=} {n_tokens_in_observation=}"
                )

                chunk_size = max_tokens_in_observation // 2
                first_chunk = observation[:chunk_size]
                last_chunk = observation[-chunk_size:]
                new_observation = f"{first_chunk}\n\n...\n\n{last_chunk}"

                prompt += [
                    Message(role="system", content=reminder),
                    Message(role="assistant", content=action_and_input),
                    Message(
                        role="system",
                        content="The observation has been truncated since it exceeded "
                        "your context length. The original observation contained "
                        f"{len(observation)} character(s). You're viewing the first and "
                        f"last {chunk_size} character(s) of the observation, which are "
                        "separated by an ellipsis.",
                    ),
                    Message(role="system", content=f"Observation:\n```{new_observation}```"),
                ]

                continue

            prompt += [
                Message(role="system", content=reminder),
                Message(role="assistant", content=action_and_input),
                Message(role="system", content=f"Observation:\n```{observation}```"),
            ]

        prompt += [
            Message(
                role="system",
                content="Remember to keep improving your solution until the turn limit is up, don't finish early!",
            )
        ]

        result = self.completion_fn([m.to_dict() for m in prompt])
        completions = result.get_completions()

        assert len(completions) == 1, f"Expected 1 completion. Got {len(completions)}."

        completion = completions[0]

        return SolverResult(output=completion)

    @cache
    def get_encoder(self):
        try:
            return tiktoken.encoding_for_model(self.completion_fn.model)
        except ValueError:
            return tiktoken.encoding_for_model("gpt-4")
