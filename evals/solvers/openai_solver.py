import logging
from typing import Any, Dict, Optional, Union

import tiktoken
from openai import BadRequestError

from evals.completion_fns.openai import OpenAIChatCompletionFn, OpenAICompletionFn
from evals.prompt.base import chat_prompt_to_text_prompt
from evals.registry import is_chat_model, n_ctx_from_model_name
from evals.solvers.solver import Solver, SolverResult
from evals.task_state import TaskState

# Default prefixes when rendering chat prompts as text
ROLE_TO_PREFIX = {
    "system": "System: ",
    "user": "User: ",
    "assistant": "Assistant: ",
    "spacer": "-----",
}


class OpenAISolver(Solver):
    """
    A solver class for OpenAI models that uses the OpenAI python SDK.

    Note: this class is also inherited by
    `evals.solvers.together_solver.TogetherSolver`, which uses the same OpenAI python
    SDK.
    """

    def __init__(
        self,
        completion_fn_options: Dict[str, Any] = {},
        valid_answers: Optional[list[str]] = None,
        fixed_start: Optional[str] = None,
        continue_last_assistant_msg: bool = False,
        role_to_prefix: Dict = ROLE_TO_PREFIX,
        postprocessors: list[str] = [],
        registry: Any = None,
    ):
        super().__init__(postprocessors=postprocessors)
        self.valid_answers = valid_answers
        self.completion_fn_options = completion_fn_options
        # Additional options for base model
        self.fixed_start = fixed_start
        self.continue_last_assistant_msg = continue_last_assistant_msg
        self.role_to_prefix = role_to_prefix

        if "model" not in completion_fn_options:
            raise ValueError("OpenAISolver requires a model to be specified.")
        model = completion_fn_options["model"]

        completion_fn_cls = self._get_completion_fn_cls(model)

        self._preprocess_completion_fn_options()

        # Create the completion function
        self.completion_fn = completion_fn_cls(
            api_base=self._api_base,
            api_key=self._api_key,
            **self.completion_fn_options,
        )

    @property
    def model(self) -> str:
        """
        Get model name from completion function, e.g. "gpt-3.5-turbo"
        This may not always include the full model version, e.g. "gpt-3.5-turbo-0613"
        so use `self.model_version` if you need the exact snapshot.
        """
        return self.completion_fn.model

    def name(self) -> str:
        return self.completion_fn.model

    @property
    def model_version(self) -> Union[str, dict]:
        """
        Makes dummy API request to get exact model version from the API
        e.g. "gpt-3.5-turbo-0613"
        """
        dummy_task_state = TaskState("", "")
        solver_result = self(dummy_task_state, **{"max_tokens": 1})
        raw_data = solver_result._metadata["raw_completion_result"].raw_data
        return raw_data.model

    def _is_chat_model(self, model: str) -> bool:
        """
        Checks in the registry if the model is a chat model.
        Implemented as a method to allow for overriding in subclasses
        (e.g. TogetherSolver, which uses a different registry of chat models)
        """
        return is_chat_model(model)

    @property
    def _completion_exception(self) -> Exception:
        """
        Returns the exception to handle when the completion function fails
        via self._handle_completion_exception
        """
        return BadRequestError

    @property
    def _api_base(self) -> Optional[str]:
        """The base URL for the API"""
        # by default, None, which points to the default API Base which is the OpenAI API
        return None

    @property
    def _api_key(self) -> Optional[str]:
        """The API key to use for the API"""
        # by default, None, which points to the default API Key which is "OPENAI_API_KEY"
        return None

    def _solve(self, task_state: TaskState, **kwargs) -> SolverResult:
        raw_msgs = [
            {"role": "system", "content": task_state.task_description},
        ] + [msg.to_dict() for msg in task_state.messages]

        precheck_outcome = self._perform_prechecks(raw_msgs)
        if precheck_outcome is not None:
            return precheck_outcome

        msgs = self._process_msgs(raw_msgs)

        try:
            if self._is_chat_model(self.model):
                completion_result = self.completion_fn(prompt=msgs, **kwargs)

                completion_output = completion_result.get_completions()[0]

                # Chat model output is already parsed, just return it
                solver_result = SolverResult(
                    completion_output, raw_completion_result=completion_result
                )
            else:
                # Manually render the prompt for completion models so that we can
                # implement things like custom render formats and/or fixed_start
                prompt = self._render_completion_prompt(msgs)

                stop_sequences = self._get_msg_separators()
                if len(stop_sequences) > 4:
                    logging.warn("Using more than 4 stop sequences is unsupported")
                completion_result = self.completion_fn(prompt=prompt, stop=stop_sequences, **kwargs)

                completion_output = completion_result.get_completions()[0]

                # Completion model output needs to be parsed to remove role prefixes
                solver_result = SolverResult(
                    self._parse_completion_response(completion_output),
                    raw_output=completion_output,
                    raw_completion_result=completion_result,
                )
        except self._completion_exception as e:
            solver_result = self._handle_completion_exception(e)

        return solver_result

    def _perform_prechecks(self, msgs: list[dict[str, str]]) -> Optional[SolverResult]:
        """
        Check if the prompt exceeds the context length before querying the
        API to avoid it contributing to the tokens per minute (TPM) limit

        If `None` is returned, the prompt is within the context length.
        """
        enc = tiktoken.encoding_for_model(self.model)
        ctx_len = n_ctx_from_model_name(self.model)
        n_tokens = 0

        for msg in msgs:
            tokens = enc.encode(msg["content"])
            n_tokens += len(tokens)

        if ctx_len is not None and n_tokens >= ctx_len:
            return SolverResult(
                output=f"Request too large for {self.model}. Context length: {ctx_len} tokens. Requested: {n_tokens} tokens.",
            )

        return None

    def _process_msgs(self, raw_msgs: list[dict[str, str]]) -> list[dict[str, str]]:
        """
        Perform any message processing before querying the API
        e.g. converting 'system' roles to 'user' roles
        """
        # By default, no message processing is performed, but subclasses can override this
        return raw_msgs

    def _handle_completion_exception(self, e: Exception) -> SolverResult:
        """
        Handles any expected exceptions from the completion function:
        - context_length_exceeded: The prompt exceeds the context length
        - too many messages: The prompt has too many messages

        Raises any other exceptions
        """
        if (
            e.code == "context_length_exceeded"
            or "Please reduce your prompt; or completion length"
            in e.message  # For context length errors where code is not specified.
        ):
            logging.warn(
                f"OpenAI API context length exceeded, using error message as solver response: {e.message}"
            )
            solver_result = SolverResult(
                e.message,
                error=e.body,
            )
        elif "'$.messages' is too long" in e.message:  # If we have too many messages
            logging.warn(
                f"Exceeded maximum chat messages on OpenAI API, using error message as solver response: {e.message}"
            )
            solver_result = SolverResult(
                e.message,
                error=e.body,
            )
        else:
            raise e

        return solver_result

    def _render_completion_prompt(self, msgs: list[dict[str, str]]) -> str:
        # Render messages as a chat dialogue in plaintext (also postfixes "Assistant: " to tee up the model)
        if self.continue_last_assistant_msg and len(msgs) > 0 and msgs[-1]["role"] == "assistant":
            self.fixed_start = msgs[-1]["content"]
            msgs = msgs[:-1]

        prompt = chat_prompt_to_text_prompt(msgs, chat_to_prefixes=self.role_to_prefix)

        # Force model to begin response with specified string
        if self.fixed_start is not None:
            prompt = prompt + " " + self.fixed_start
        return prompt

    def _parse_completion_response(self, raw_response: str) -> str:
        # Parse response up to the next message separator
        # e.g. "System:", "User:", "Assistant:", "-----"
        msg_separators = self._get_msg_separators()

        parsed_response = raw_response
        for msg_sep in msg_separators:
            parsed_response = parsed_response.split(msg_sep)[0].strip()

        # The fixed_start should be included in the response
        if self.fixed_start is not None:
            parsed_response = self.fixed_start + " " + parsed_response
        return parsed_response

    def _get_msg_separators(self) -> list[str]:
        """Return the separators between parts of the prompt (e.g. "User:", "-----").

        This is used to cut hallucination from base models.
        """
        return [v.strip() for v in self.role_to_prefix.values() if v.strip() != ""]

    def _get_completion_fn_cls(self, model: str) -> Any:
        # Infer suitable CompletionFn class from the model name
        if self._is_chat_model(model):
            completion_fn_cls = OpenAIChatCompletionFn
            if self.fixed_start is not None or self.continue_last_assistant_msg:
                raise ValueError(
                    "OpenAISolver does not support fixed_start or continue_last_assistant_msg with chat models."
                )
        else:
            if self.fixed_start is not None and self.continue_last_assistant_msg:
                raise ValueError(
                    "OpenAISolver does not support both fixed_start and continue_last_assistant_msg being used."
                )

            completion_fn_cls = OpenAICompletionFn

        return completion_fn_cls

    def _preprocess_completion_fn_options(self) -> dict:
        """
        Preprocess the completion function options before creating the completion function

        e.g. apply logit biasing
        """
        model = self.completion_fn_options["model"]
        # If valid answers were provided, apply logit bias to those tokens
        if self.valid_answers is not None and len(self.valid_answers) > 0:
            self.completion_fn_options["extra_options"]["logit_bias"] = self._make_logit_bias(
                self.valid_answers, model
            )

    def _make_logit_bias(self, valid_answers: list[str], model: str) -> dict[int, float]:
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
