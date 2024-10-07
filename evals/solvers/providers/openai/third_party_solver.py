import os
from typing import Optional

from openai import PermissionDeniedError

from evals.solvers.providers.openai.openai_solver import OpenAISolver
from evals.solvers.solver import SolverResult


class ThirdPartySolver(OpenAISolver):
    """
    A solver class for third-party inference providers that expose an OpenAI-compatible API.
    Leverages the OpenAISolver class, with some overrides.

    Specifically we override:
    - `_api_base` to point to the actual API base URL
    - `_api_key` to use the appropriate API key
    - `_is_chat_model` to return True, since most (all?) third-party APIs support chat models
    - `_preprocess_completion_fn_options` to not perform any completion fn options preprocessing
    - `_perform_prechecks` to not perform any checks before calling the API

    Additionally, the `valid_answers` parameter is not supported by third-party APIs, so we raise an error if it's passed in.
    """

    def __init__(self, api_base: str, api_key_env_var: str, **kwargs):
        self.api_base = api_base
        self.api_key = kwargs.pop("api_key", None) or os.environ.get(api_key_env_var)
        super().__init__(**kwargs)
        if self.valid_answers is not None:
            raise NotImplementedError("`valid_answers` not supported by ThirdPartySolver")

    @property
    def _api_base(self) -> Optional[str]:
        """The base URL for the API"""
        return self.api_base

    @property
    def _api_key(self) -> Optional[str]:
        """The API key to use for the API"""
        return self.api_key

    @property
    def _completion_exception(self) -> Exception:
        """
        Overrides OpenAISolver implementation;
        ThirdPartySolver API uses a different error code to signal context length issues
        """
        return PermissionDeniedError

    def _is_chat_model(self, model: str) -> bool:
        """
        Overrides OpenAISolver implementation; assumes all third-party APIs only support chat models
        """
        return True

    def _preprocess_completion_fn_options(self) -> dict:
        """
        Overrides OpenAISolver implementation; Here we do not perform any completion fn
        options preprocessing since the ThirdPartySolver does not support the
        `valid_answers` parameter
        """

    def _perform_prechecks(self, msgs: list[dict[str, str]]) -> Optional[SolverResult]:
        """
        Overrides OpenAISolver implementation; Here we do not perform any prechecks
        since the ThirdPartySolver does not support context length checks due to the lack
        of a tokenizer.
        """
        return None
