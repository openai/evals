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

    def __init__(self, api_base: Optional[str] = None, **kwargs):
        self.api_base = api_base
        super().__init__(**kwargs)
        if self.valid_answers is not None:
            raise NotImplementedError("`valid_answers` not supported by FixieSolver")

    @property
    def _api_base(self) -> Optional[str]:
        """The base URL for the API"""
        return self.api_base or "https://ultravox.api.fixie.ai/v1"

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
