import json
import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, TypeVar, Union

import tiktoken

from evals.api import CompletionFn
from evals.task_state import TaskState

SolverType = TypeVar("SolverType", bound="Solver")


class SolverResult:
    def __init__(self, output: str, **metadata):
        self._output = output
        self._metadata = metadata

    @property
    def output(self) -> str:
        return self._output

    @property
    def metadata(self) -> dict:
        return self._metadata

    def to_json(self) -> str:
        return json.dumps(
            {
                "output": self.output,
                **self.metadata,
            },
            indent=2,
        )


class Solver(ABC, CompletionFn):
    # We need to inherit from CompletionFn because of how the oaival registry works.

    @abstractmethod
    def __call__(
        self,
        task_state: TaskState,
        **kwargs,
    ) -> SolverResult:
        """
        ARGS
        ====
        `task_state`: A `TaskState` object that contains the task description and the input.
        `kwargs`: Other arguments passed to the solver.

        RETURNS
        =======
        The result of the solver.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Name of the Solver. This is intended mostly for logging.

        RETURNS
        =======
        A human-readable name that describes this solver.
        """

    def copy(self: SolverType) -> SolverType:
        #   The deepcopy may be quite heavy for some solvers; if that's the
        #   case they should override this function.
        return deepcopy(self)


class OpenAISolver(Solver):
    """An abstract solver class that uses the OpenAI API through completion functions."""

    def __init__(
        self,
        completion_fn_options: Dict[str, Any] = {},
        valid_answers: Union[list[str], None] = None,
    ):
        self.completion_fn_options = completion_fn_options

        # If valid answers were provided, encode them into a logit bias dictionary.
        if valid_answers is not None and len(valid_answers) > 0:
            model = completion_fn_options["model"] if "model" in completion_fn_options else None
            if model is None:
                raise ValueError("OpenAISolver requires a model to be specified.")
            if model == "code-davinci-002":
                logging.info(
                    f"Attempting to use logit bias with model {model}, which does not support logit bias."
                )

            enc = tiktoken.encoding_for_model(model)
            token_ids = []
            for answer in valid_answers:
                encoded_answer = enc.encode(answer)
                if len(encoded_answer) > 1:
                    raise ValueError(
                        f"Answer {answer} was encoded to {encoded_answer}, but we expected a single token."
                    )
                token_ids.append(encoded_answer[0])
            self.completion_fn_options["extra_options"]["logit_bias"] = {
                token_id: 100 for token_id in token_ids
            }


class DummySolver(Solver):
    def __call__(
        self,
        task_state: TaskState,
        **kwargs,
    ) -> SolverResult:
        return SolverResult("This is a dummy response.")

    @property
    def name(self) -> str:
        return "DummySolver"
