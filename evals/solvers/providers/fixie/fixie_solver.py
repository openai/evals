import os
from typing import Optional

from evals.solvers.providers.openai.openai_solver import OpenAISolver
from evals.solvers.solver import SolverResult


def is_chat_model(model: str) -> bool:
    if model.startswith("fixie-ai/"):
        return True
    else:
        raise NotImplementedError(f"Model {model} not currently supported by FixieSolver")


class FixieSolver(OpenAISolver):
    """
    A solver class for the Fixie Ultravox API via the OpenAI python SDK completion functions.
    Leveraging the OpenAISolver class, with some overrides.

    Specifically we override:
    - `_api_base` to point to the Ultravox API server
    - `_api_key` to use the Ultravox_API_KEY environment variable
    - `_is_chat_model` to use a different dictionary of supported chat models
    - `_perform_prechecks` to not try to calculate token lengths on audio inputs
    """

    def __init__(self, merge_adjacent_msgs: bool = False, **kwargs):
        os.environ["EVALS_SEQUENTIAL"] = "1"  # Needed until vLLM supports parallel requests
        super().__init__(**kwargs)
        self.merge_adjacent_msgs = merge_adjacent_msgs
        if self.valid_answers is not None:
            raise NotImplementedError("`valid_answers` not supported by FixieSolver")

    @property
    def _api_base(self) -> Optional[str]:
        """The base URL for the API"""
        return "https://ultravox.api.fixie.ai/v1"

    @property
    def _api_key(self) -> Optional[str]:
        """The API key to use for the API"""
        return os.environ.get("ULTRAVOX_API_KEY")

    def _is_chat_model(self, model: str) -> bool:
        """
        Overrides OpenAISolver implementation;
        Need to use different dictionary of chat models
        """
        return is_chat_model(model)

    def _perform_prechecks(self, msgs: list[dict[str, str]]) -> Optional[SolverResult]:
        """
        Overrides OpenAISolver implementation; Here we do not perform any prechecks
        since the TogetherSolver does not support context length checks due to the lack
        of a tokenizer.
        """
        return None

    #  def _process_msgs(self, msgs: list[dict[str, str]]) -> list[dict[str, str]]:
    #     """
    #     Many OS models, like Llama-2 and Mixtral, expect a more specific format than
    #     we often provide to OpenAI models. In particular
    #     - there should only be a single system prompt, at the start
    #     - there should be at least one user prompt
    #     - after an optional system prompt, the messages should alternate between
    #         user and assistant messages.
    #     """
    #     msgs = copy.deepcopy(msgs)

    #     # if there is only a system message, turn it to a user message
    #     if len(msgs) == 1 and msgs[0]["role"] == "system":
    #         return [{"role": "user", "content": msgs[0]["content"]}]

    #     # convert all system messages except a possible first one to user messages
    #     for i, msg in enumerate(msgs):
    #         if msg["role"] == "system" and i > 0:
    #             msg["role"] = "user"

    #     # if the first message is a system message and the second one is an assistant message,
    #     # this implies that we previously converted the initial system message to a user message,
    #     # so we should convert the initial system message to a user message again for consistency
    #     # NOTE: this looks like it'd fail on length 1 messages, but that's handled by the first if
    #     # combined with the first statement of this if and lazy evaluation
    #     if msgs[0]["role"] == "system" and msgs[1]["role"] == "assistant":
    #         msgs[0]["role"] = "user"

    #     # before returning, we optionally merge all adjacent messages from the same role
    #     if self.merge_adjacent_msgs:
    #         merged_msgs = []
    #         for msg in msgs:
    #             if len(merged_msgs) > 0 and merged_msgs[-1]["role"] == msg["role"]:
    #                 merged_msgs[-1]["content"] += "\n\n" + msg["content"]
    #             else:
    #                 merged_msgs.append(msg)
    #         msgs = merged_msgs
    #     return msgs
