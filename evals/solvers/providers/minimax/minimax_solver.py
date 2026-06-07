import logging
import os
from typing import Optional

from openai import BadRequestError

from evals.solvers.providers.openai.openai_solver import OpenAISolver
from evals.solvers.solver import SolverResult

MINIMAX_CHAT_MODELS = {
    "MiniMax-M3",
    "MiniMax-M2.7",
    "MiniMax-M2.7-highspeed",
}


def is_chat_model(model: str) -> bool:
    if model in MINIMAX_CHAT_MODELS:
        return True
    raise NotImplementedError(f"Model {model} not currently supported by MiniMaxSolver")


class MiniMaxSolver(OpenAISolver):
    """
    A solver class for the MiniMax API via the OpenAI python SDK completion functions.
    Leveraging the OpenAISolver class, with some overrides.

    Specifically we override:
    - `_api_base` to point to the MiniMax API (OpenAI-compatible endpoint)
    - `_api_key` to use the MINIMAX_API_KEY environment variable
    - `_is_chat_model` to use MiniMax's supported chat models
    - `_preprocess_completion_fn_options` to not perform any completion fn options preprocessing
    - `_perform_prechecks` to not perform any checks before calling the API

    MiniMax API constraints:
    - temperature must be in (0.0, 1.0], cannot be 0
    - response_format is not supported
    - valid_answers (logit_bias) is not supported
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.valid_answers is not None:
            raise NotImplementedError("`valid_answers` not supported by MiniMaxSolver")

    @property
    def _api_base(self) -> Optional[str]:
        """The base URL for the API"""
        return os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.io/v1")

    @property
    def _api_key(self) -> Optional[str]:
        """The API key to use for the API"""
        return os.environ.get("MINIMAX_API_KEY")

    @property
    def _completion_exception(self) -> Exception:
        """
        Overrides OpenAISolver implementation;
        MiniMax API uses BadRequestError for context length issues
        """
        return BadRequestError

    def _is_chat_model(self, model: str) -> bool:
        """
        Overrides OpenAISolver implementation;
        Need to use different dictionary of chat models
        """
        return is_chat_model(model)

    def _preprocess_completion_fn_options(self) -> dict:
        """
        Overrides OpenAISolver implementation; Here we do not perform any completion fn
        options preprocessing since the MiniMaxSolver does not support the
        `valid_answers` parameter (logit_bias)
        """

    def _perform_prechecks(self, msgs: list[dict[str, str]]) -> Optional[SolverResult]:
        """
        Overrides OpenAISolver implementation; Here we do not perform any prechecks
        since the MiniMaxSolver does not support context length checks due to the lack
        of a tokenizer in tiktoken for MiniMax models.
        """
        return None

    def _handle_completion_exception(self, e: Exception) -> SolverResult:
        """
        Handles any expected exceptions from the MiniMax API completion function.
        """
        if hasattr(e, "code") and e.code == "context_length_exceeded":
            logging.warning(
                f"MiniMax API context length exceeded, using error message as solver response: {e.message}"
            )
            solver_result = SolverResult(
                e.message,
                error=e.body,
            )
        elif hasattr(e, "message") and (
            "Please reduce your prompt" in e.message
            or "'$.messages' is too long" in e.message
        ):
            logging.warning(
                f"MiniMax API error, using error message as solver response: {e.message}"
            )
            solver_result = SolverResult(
                e.message,
                error=e.body,
            )
        else:
            raise e

        return solver_result
