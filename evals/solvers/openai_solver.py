import logging
from typing import Any, Dict, Optional

import tiktoken
from openai import BadRequestError

from evals.completion_fns.openai import OpenAIChatCompletionFn, OpenAICompletionFn
from evals.prompt.base import chat_prompt_to_text_prompt
from evals.registry import is_chat_model
from evals.solvers.solver import Solver, SolverResult
from evals.task_state import TaskState


class OpenAISolver(Solver):
    """A solver class that uses the OpenAI API through completion functions."""

    def __init__(
        self,
        completion_fn_options: Dict[str, Any] = {},
        valid_answers: Optional[list[str]] = None,
        fixed_start: Optional[str] = None,
        registry: Any = None,
    ):
        self.completion_fn_options = completion_fn_options
        self.fixed_start = fixed_start

        if "model" not in completion_fn_options:
            raise ValueError("OpenAISolver requires a model to be specified.")
        model = completion_fn_options["model"]

        # Infer suitable CompletionFn class from the model name
        if is_chat_model(model):
            completion_fn_cls = OpenAIChatCompletionFn
            if self.fixed_start is not None:
                raise ValueError("OpenAISolver does not support fixed_start with chat models.")
        else:
            completion_fn_cls = OpenAICompletionFn

        # If valid answers were provided, apply logit bias to those tokens
        if valid_answers is not None and len(valid_answers) > 0:
            self.completion_fn_options["extra_options"]["logit_bias"] = self._make_logit_bias(
                valid_answers,
                model,
            )

        # Create the completion function
        self.completion_fn = completion_fn_cls(
            **self.completion_fn_options,
        )

    @property
    def model(self) -> str:
        return self.completion_fn.model

    @property
    def is_completion_model(self) -> bool:
        return not is_chat_model(self.model)

    def _make_logit_bias(self, valid_answers: list[str], model: str) -> dict[int, float]:
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
        return {token_id: 100 for token_id in token_ids}

    def _render_completion_prompt(self, msgs: list[dict[str, str]]) -> str:
        # Render messages as a chat dialogue in plaintext (also postfixes "Assistant: " to tee up the model)
        prompt = chat_prompt_to_text_prompt(msgs)

        # Force model to begin response with fixed_start
        if self.fixed_start is not None:
            prompt = prompt + self.fixed_start
        return prompt

    def _parse_completion_response(self, raw_response: str) -> str:
        # Parse response up to the next message separator
        # Technically should look for new messages from "system" role too, but
        # the default renderer doesn't show a prefix for new system messages.
        msg_separators = ["User:", "Assistant:", "-----"]

        parsed_response = raw_response
        for msg_sep in msg_separators:
            parsed_response = parsed_response.split(msg_sep)[0].strip()

        # The fixed_start should be included in the response
        if self.fixed_start is not None:
            parsed_response = self.fixed_start + parsed_response
        return parsed_response

    def _solve(
        self,
        task_state: TaskState,
        **kwargs,
    ) -> SolverResult:

        msgs = [
            {"role": "system", "content": task_state.task_description},
        ] + [msg.to_dict() for msg in task_state.messages]

        try:
            if self.is_completion_model:
                # Manually render the prompt for completion models so that we can
                # implement things like custom render formats and/or fixed_start
                prompt = self._render_completion_prompt(msgs)
                completion_result = self.completion_fn(prompt=prompt, **kwargs)

                completion_output = completion_result.get_completions()[0]

                # Completion model output needs to be parsed to remove role prefixes
                solver_result = SolverResult(
                    self._parse_completion_response(completion_output),
                    raw_output=completion_output,
                )
            else:
                completion_result = self.completion_fn(prompt=msgs, **kwargs)

                completion_output = completion_result.get_completions()[0]

                # Chat model output is already parsed, just return it
                solver_result = SolverResult(completion_output)
        except BadRequestError as e:
            if (
                e.code == "context_length_exceeded"
                or "Please reduce your prompt; or completion length" in e.message   # For context length errors where code is not specified.
            ):
                logging.warn(
                    f"OpenAI API context length exceeded, using error message as solver response: {e.message}"
                )
                solver_result = SolverResult(
                    e.message,
                    error=e.body,
                )
            else:
                raise e
        return solver_result

    @property
    def name(self) -> str:
        return self.completion_fn.model
