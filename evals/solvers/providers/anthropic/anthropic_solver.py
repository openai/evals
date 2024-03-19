from typing import Any, Optional, Union

from evals.solvers.solver import Solver, SolverResult
from evals.task_state import TaskState, Message
from evals.record import record_sampling
from evals.utils.api_utils import request_with_timeout

import anthropic
from anthropic import Anthropic
from anthropic.types import ContentBlock, MessageParam, Usage
import backoff

oai_to_anthropic_role = {
    "system": "user",
    "user": "user",
    "assistant": "assistant",
}


class AnthropicSolver(Solver):
    """
    A solver class that uses the Anthropic API for textual chat-based tasks.
    """

    def __init__(
        self,
        model_name: str,
        max_tokens: int = 512,
        postprocessors: list[str] = [],
        extra_options: Optional[dict] = {},
        registry: Any = None,
    ):
        super().__init__(postprocessors=postprocessors)
        # https://docs.anthropic.com/claude/docs/models-overview#model-comparison
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.extra_options = extra_options

    def _solve(self, task_state: TaskState, **kwargs) -> SolverResult:
        """
        Solve the task using the Anthropic API
        """
        orig_msgs = task_state.messages
        anth_msgs = self._convert_msgs_to_anthropic_format(task_state.messages)

        # TODO: handle context length limit; possible once anthropic tokenizer is available

        # calls client.messages.create, but is wrapped with backoff retrying decorator
        response = anthropic_create_retrying(
            client=Anthropic(max_retries=0),  # we take care of retries ourselves
            model=self.model_name,
            system=task_state.task_description,
            messages=anth_msgs,
            max_tokens=self.max_tokens,  # required kwarg for messages.create
            **{**kwargs, **self.extra_options},
        )
        solver_result = SolverResult(
            output=response.content[0].text, raw_completion_result=response.content
        )

        # for logging purposes: prepend the task desc to the orig msgs as a system message
        orig_msgs.insert(
            0, Message(role="system", content=task_state.task_description).to_dict()
        )
        record_sampling(
            prompt=orig_msgs,  # original message format, supported by our logviz
            sampled=[solver_result.output],
            model=self.model_name,
            usage=anth_to_openai_usage(response.usage),
        )
        return solver_result

    @property
    def name(self) -> str:
        return self.model_name

    @property
    def model_version(self) -> Union[str, dict]:
        """
        For the moment, Anthropic does not use aliases,
        so model_version is the same as model_name.
        """
        return self.model_name

    @staticmethod
    def _convert_msgs_to_anthropic_format(msgs: list[Message]) -> list[MessageParam]:
        """
        Anthropic API requires that the message list has
        - Roles as 'user' or 'assistant'
        - Alternating 'user' and 'assistant' messages

        Note: the top-level system prompt is handled separately and should not be
        included in the messages list.
        """
        # enforce valid roles; convert to Anthropic message type
        anth_msgs = [
            MessageParam(
                role=oai_to_anthropic_role[msg.role],
                content=[ContentBlock(text=msg.content, type="text")],
            )
            for msg in msgs
        ]
        # enforce alternating roles by merging consecutive messages with the same role
        # e.g. [user1, user2, assistant1, user3] -> [user12, assistant1, user3]
        alt_msgs = []
        for msg in anth_msgs:
            if len(alt_msgs) > 0 and msg["role"] == alt_msgs[-1]["role"]:
                # Merge consecutive messages from the same role
                alt_msgs[-1]["content"].extend(msg["content"])
            else:
                alt_msgs.append(msg)

        return alt_msgs


@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=(
        anthropic.RateLimitError,
        anthropic.APIConnectionError,
        anthropic.APITimeoutError,
        anthropic.InternalServerError,
    ),
    max_value=60,
    factor=1.5,
)
def anthropic_create_retrying(client: Anthropic, *args, **kwargs):
    """
    Helper function for creating a backoff-retry enabled message request.
    `args` and `kwargs` match what is accepted by `client.messages.create`.
    """
    result = request_with_timeout(client.messages.create, *args, **kwargs)
    if "error" in result:
        raise Exception(result["error"])
    return result


def anth_to_openai_usage(anth_usage: Usage) -> dict:
    """
    Processes anthropic Usage object into dict with keys
    that match the OpenAI Usage dict, for logging purposes.
    """
    # TODO: make this format of dict a dataclass type to be reused througout lib?
    return {
        "completion_tokens": anth_usage.output_tokens,
        "prompt_tokens": anth_usage.input_tokens,
        "total_tokens": anth_usage.input_tokens + anth_usage.output_tokens,
    }
