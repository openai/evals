import copy
import logging
import re
from functools import cached_property
from typing import Any

import tiktoken

from evals.completion_fns.openai import OpenAIChatCompletionFn
from evals.elsuite.multistep_web_tasks.solvers.strong_solver.strong_prompts import (
    EXAMPLE_TEMPLATE,
    PROMPT,
)
from evals.elsuite.multistep_web_tasks.utils import MWTTaskState
from evals.prompt.base import OpenAICreateChatPrompt
from evals.registry import is_chat_model, n_ctx_from_model_name
from evals.solvers.solver import Solver, SolverResult
from evals.task_state import Message

logger = logging.getLogger(__name__)
# 2048 is the number of tokens for the old gpt-3 models, so is a decent lower bound
MINIMUM_CONTEXT_LENGTH = 2048
# There are some mandatory tokens associated with each message
# I'll use 4 to be slightly conservative
TOKENS_PER_MESSAGE = 4
# A small buffer to avoid exceeding the context length by a few tokens
TOKEN_BUFFER = 10


class StrongSolver(Solver):
    """Chat-model-based solver that uses Chain of Thought by default."""

    def __init__(
        self,
        completion_fn_options: dict[str, Any] = {},
        action_splitter: str = "```",
        **kwargs,
    ):
        # NOTE: assumes a chat completion fn
        assert is_chat_model(
            completion_fn_options["model"]
        ), f"StrongSolver needs a chat model, got {completion_fn_options['model']}"
        self.completion_fn = OpenAIChatCompletionFn(
            **completion_fn_options,
        )

        self.max_response_tokens = completion_fn_options["extra_options"].get("max_tokens")
        if self.max_response_tokens is None:
            raise ValueError("Must set max_tokens in yaml to avoid exceeding context length")

        self.context_length = self._get_context_length()

        self.action_splitter = action_splitter

    @cached_property
    def encoding(self) -> tiktoken.Encoding:
        # we use a cached property here to avoid having to pickle the encoding
        # (so that deepcopy works in SolverEval)
        return self._get_encoding()

    def _get_encoding(self) -> tiktoken.Encoding:
        model = self.completion_fn.model
        assert model is not None
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            logger.warning(
                f"Warning: tokenizer for '{model}' not found. Using cl100k_base encoding."
            )
            encoding = tiktoken.get_encoding("cl100k_base")
        return encoding

    def _get_context_length(self) -> int:
        assert self.completion_fn.model is not None
        n_ctx = n_ctx_from_model_name(self.completion_fn.model)

        context_length = n_ctx if n_ctx is not None else MINIMUM_CONTEXT_LENGTH
        logger.info(
            f"Model {self.completion_fn.model} has n_ctx={n_ctx} and max_tokens={self.max_response_tokens}"
        )
        return context_length

    def _solve(
        self,
        task_state: MWTTaskState,
        **kwargs,
    ) -> SolverResult:
        base_prompt = PROMPT.format(action_splitter=self.action_splitter)
        current_example_template = EXAMPLE_TEMPLATE

        # TODO: use as many previous observations as will fit in the context, rather than just 3
        new_observation = self._get_new_observation_from_task_state(task_state)
        previous_action = self._get_previous_action_from_task_state(task_state)
        current_example = current_example_template.format(
            observation=new_observation,
            previous_action=previous_action,
            # remnants of previous WebArena implementation
            objective=task_state.goal,
            url=task_state.url if task_state.url else "None",
        )
        truncated_messages = task_state.messages[:-1]  # last message is handled separately
        modified_messages = self._add_action_splitter_to_actions(truncated_messages)
        messages: OpenAICreateChatPrompt = [
            {"role": "system", "content": base_prompt},
            *[msg.to_dict() for msg in modified_messages],
            {"role": "user", "content": current_example},
        ]

        final_messages = self._cut_messages_to_fit(messages)
        response = self.completion_fn(final_messages)
        parsed_action = self._extract_action(response.get_completions()[0])
        return SolverResult(parsed_action)

    def _add_action_splitter_to_actions(self, messages: list[Message]) -> list[Message]:
        """To avoid gpt-3.5 (and gpt-4) getting too confused, I'll make it so
        the previous actions in the trajectory are rendered with the action
        splitter (but sadly still not the chain of thought)"""
        new_message_list = []
        for message in messages:
            if message.role == "assistant":
                message = copy.deepcopy(message)
                message.content = f"{self.action_splitter}{message.content}{self.action_splitter}"
            new_message_list.append(message)
        return new_message_list

    def _cut_messages_to_fit(self, messages: OpenAICreateChatPrompt) -> OpenAICreateChatPrompt:
        """Remove messages from the prompt, starting with the first observation,
        until it fits within the context window"""
        target_n_tokens = self.context_length - self.max_response_tokens - TOKEN_BUFFER
        logger.debug(f"{target_n_tokens = }")
        messages_tokens = [self.encoding.encode(msg["content"]) for msg in messages]
        messages_n_tokens = [len(tokens) + TOKENS_PER_MESSAGE for tokens in messages_tokens]
        total_n_tokens = sum(messages_n_tokens)
        logger.debug(f"{total_n_tokens = }")

        if total_n_tokens < target_n_tokens:
            logger.debug("initial prompt is short enough, returning!")
            return messages

        if len(messages) < 2:
            raise ValueError("Not enough messages (only 1, which is system)")

        # try to cut messages to get below the target tokens
        if len(messages) > 2:
            for i in range(1, len(messages) - 1):
                logger.debug(f"truncating messages, {i = }, {total_n_tokens = }")
                logger.debug(f"{len(messages) = }, [:1] and [{i} + 1:]")
                if total_n_tokens < target_n_tokens:
                    return messages[:1] + messages[i + 1 :]
                total_n_tokens -= messages_n_tokens[i]
        # if after the loop we didn't succeed, just take the first and last messages
        remaining_messages = messages[:1] + messages[-1:]

        if len(remaining_messages) != 2:
            logger.debug(f"{len(remaining_messages) = }")
            logger.debug(f"{[msg['role'] for msg in remaining_messages] = }")
        assert len(remaining_messages) == 2, "At this point, should only be two messages left"

        # only one observation (and system message), so we have to shorten the obs rather than drop it
        messages = copy.deepcopy(remaining_messages)

        token_budget_for_obs = target_n_tokens - messages_n_tokens[0]
        truncated_content_tokens = messages_tokens[-1][:token_budget_for_obs]
        truncated_content_text = self.encoding.decode(truncated_content_tokens)
        untruncated_content_text = messages[-1]["content"]
        logger.debug(f"{len(untruncated_content_text) = }")
        logger.debug(f"{len(truncated_content_text) = }")
        logger.debug(f"{len(truncated_content_tokens) = }")
        logger.debug(
            f"final total length = {len(truncated_content_tokens) + messages_n_tokens[0] = }"
        )
        remaining_messages[1]["content"] = f"OBSERVATION: {truncated_content_text}"
        return messages

    def _get_new_observation_from_task_state(self, task_state: MWTTaskState) -> str:
        new_observation = task_state.messages[-1].content
        return new_observation

    def _get_previous_action_from_task_state(self, task_state: MWTTaskState) -> str:
        if len(task_state.messages) < 2:
            # so far there's only one observation and no previous action
            return "None"
        else:
            return task_state.messages[-2].content

    def _extract_action(self, response: str) -> str:
        logger.info(f"Extracting action from response:\n{response}")
        action_splitter = self.action_splitter
        pattern = rf"{action_splitter}(.*?){action_splitter}"
        match = re.search(pattern, response)
        if match:
            return match.group(1)
        else:
            logger.warn(
                f"Cannot parse action from response:\n[[{response}]]\nReturning raw response"
            )
            return response

    def name(self) -> str:
        return "StrongSolver"


# some testing
def main():
    completion_fn_options = {
        # "model": "gpt-4-32k-0613",
        "model": "gpt-3.5-turbo-16k-0613",
        "extra_options": {
            "max_tokens": 200,
        },
    }
    solver = StrongSolver(completion_fn_options)
    messages = [
        Message(role="system", content="This is a really long system message." "" * 200),
        *[Message(role="user", content="This is a shorter user message" * 100) for i in range(100)],
        Message(
            role="user", content="OBSERVATION: " + "This is a really long final message" * 10000
        ),
    ]
    chat_prompt: OpenAICreateChatPrompt = [msg.to_dict() for msg in messages]
    final_messages = solver._cut_messages_to_fit(chat_prompt)
    assert len(final_messages) == 2


if __name__ == "__main__":
    main()
