import os
from typing import Optional

from openai import PermissionDeniedError

from evals.solvers.providers.openai.openai_solver import OpenAISolver
from evals.solvers.solver import SolverResult


def is_chat_model(model: str) -> bool:
    # NOTE: this is just as brittle as evals.registry.is_chat_model
    # that we use for OpenAI models
    if True:  # model in {"accounts/fireworks/models/llama-v3p1-8b-instruct"}:
        return True
    elif model in {}:
        return False
    else:
        raise NotImplementedError(f"Model {model} not currently supported by MistralSolver")


class MistralSolver(OpenAISolver):
    """
    A solver class for the Mistral API via the OpenAI python SDK completion functions.
    Leveraging the OpenAISolver class, with some overrides.

    Specifically we override:
    - `_api_base` to point to the Mistral API
    - `_api_key` to use the MISTRAL_API_KEY environment variable
    - `_is_chat_model` to use a different dictionary of supported chat models
    - `_preprocess_completion_fn_options` to not perform any completion fn options preprocessing
    - `_perform_prechecks` to not perform any checks before calling the API

    Additionally, the `valid_answers` parameter is not supported by the Mistral API
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.valid_answers is not None:
            raise NotImplementedError("`valid_answers` not supported by MistralSolver")

    @property
    def _api_base(self) -> Optional[str]:
        """The base URL for the API"""
        return "https://api.mistral.ai/v1"

    @property
    def _api_key(self) -> Optional[str]:
        """The API key to use for the API"""
        return os.environ.get("MISTRAL_API_KEY")

    @property
    def _completion_exception(self) -> Exception:
        """
        Overrides OpenAISolver implementation;
        MistralSolver API uses a different error code to signal context length issues
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
        options preprocessing since the MistralSolver does not support the
        `valid_answers` parameter
        """

    def _perform_prechecks(self, msgs: list[dict[str, str]]) -> Optional[SolverResult]:
        """
        Overrides OpenAISolver implementation; Here we do not perform any prechecks
        since the MistralSolver does not support context length checks due to the lack
        of a tokenizer.
        """
        return None
