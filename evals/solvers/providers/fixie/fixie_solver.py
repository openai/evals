import copy
import logging
import os
from typing import Optional

from openai import PermissionDeniedError

from evals.solvers.providers.openai.openai_solver import OpenAISolver
from evals.solvers.solver import SolverResult


def is_chat_model(model: str) -> bool:
    if model.startswith("fixie-ai/"):
        return True
    else:
        raise NotImplementedError(f"Model {model} not currently supported by TogetherSolver")


class FixieSolver(OpenAISolver):
    """
    A solver class for the Together API via the OpenAI python SDK completion functions.
    Leveraging the OpenAISolver class, with some overrides.

    Specifically we override:
    - `_api_base` to point to the Together API
    - `_api_key` to use the TOGETHER_API_KEY environment variable
    - `_is_chat_model` to use a different dictionary of supported chat models
    - `_preprocess_completion_fn_options` to not perform any completion fn options preprocessing
    - `_perform_prechecks` to not perform any checks before calling the API
    - `_process_msgs` to convert message roles to comply with the Together API
    - `_completion_exception` to use the Together API's error code for context length
    - `_handle_completion_exception` to handle Together API errors differently

    Additionally, the `valid_answers` parameter is not supported by the Together API
    """

    def __init__(self, merge_adjacent_msgs: bool = False, **kwargs):
        os.environ["EVALS_SEQUENTIAL"] = "1"
        super().__init__(**kwargs)
        self.merge_adjacent_msgs = merge_adjacent_msgs
        if self.valid_answers is not None:
            raise NotImplementedError("`valid_answers` not supported by TogetherSolver")

    @property
    def _api_base(self) -> Optional[str]:
        """The base URL for the API"""
        return "https://ultravox.api.fixie.ai/v1"

    @property
    def _api_key(self) -> Optional[str]:
        """The API key to use for the API"""
        return os.environ.get("ULTRAVOX_API_KEY")

    @property
    def _completion_exception(self) -> Exception:
        """
        Overrides OpenAISolver implementation;
        Together API uses a different error code to signal context length issues
        """
        return PermissionDeniedError

    def _is_chat_model(self, model: str) -> bool:
        """
        Overrides OpenAISolver implementation;
        Need to use different dictionary of chat models
        """
        return is_chat_model(model)

    def _preprocess_completion_fn_options(self) -> dict:
        """
        Overrides OpenAISolver implementation; Here we do not perform any completion fn
        options preprocessing since the TogetherSolver does not support the
        `valid_answers` parameter
        """

    def _perform_prechecks(self, msgs: list[dict[str, str]]) -> Optional[SolverResult]:
        """
        Overrides OpenAISolver implementation; Here we do not perform any prechecks
        since the TogetherSolver does not support context length checks due to the lack
        of a tokenizer.
        """
        return None

    def _process_msgs(self, msgs: list[dict[str, str]]) -> list[dict[str, str]]:
        """
        Many OS models, like Llama-2 and Mixtral, expect a more specific format than
        we often provide to OpenAI models. In particular
        - there should only be a single system prompt, at the start
        - there should be at least one user prompt
        - after an optional system prompt, the messages should alternate between
            user and assistant messages.
        """
        return msgs

    def _handle_completion_exception(self, e: Exception) -> SolverResult:
        """
        Overrides OpenAISolver implementation; TogetherSolver is a bit less granular
        and the errors are parsed differently.
        """
        if e.type == "invalid_request_error":
            logging.warn(f"Together API context length exceeded, using error message as solver response: {e.message}")
            solver_result = SolverResult(
                e.message,
                error=e.body,
            )
        else:
            raise e

        return solver_result
