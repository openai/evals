"""
These Solvers are modified versions of the agents implemented in the original
WebArena project. The modifications include adding a Bash command, and editing
the instructions a little (such as replacing one Browser few-shot example with a
Bash few-shot example).
"""

import logging
import re
from typing import Any

from evals.completion_fns.openai import OpenAIChatCompletionFn
from evals.elsuite.multistep_web_tasks.solvers.webarena_solvers.webarena_prompts import (
    COT_BASH_BROWSER_PROMPT,
    COT_BROWSER_PROMPT,
)
from evals.elsuite.multistep_web_tasks.utils import MWTTaskState
from evals.prompt.base import OpenAICreateChatPrompt
from evals.solvers.solver import Solver, SolverResult

logger = logging.getLogger(__name__)


class WebArenaSolver(Solver):
    """Rewriting the WebArena Agent here because
    it's too messy to try to wrap it"""

    def __init__(
        self,
        completion_fn_options: dict[str, Any] = {},
        action_splitter: str = "```",
        **kwargs,
    ):
        # NOTE: assumes a chat completion fn
        self.completion_fn = OpenAIChatCompletionFn(
            **completion_fn_options,
        )
        self.action_splitter = action_splitter

    def __call__(
        self,
        task_state: MWTTaskState,
        **kwargs,
    ) -> SolverResult:
        raise NotImplementedError

    def extract_action(self, response: str) -> str:
        logger.info(f"\nExtracting action from response:\n{response}\n=====\n")
        action_splitter = self.action_splitter
        pattern = rf"{action_splitter}(.*?){action_splitter}"
        match = re.search(pattern, response)
        if match:
            return match.group(1)
        else:
            logger.warn(
                f"\nCannot parse action from response: \n[[{response}]]\n, returning raw response\n=====\n"
            )
            return response


class BrowserWebArenaSolver(WebArenaSolver):
    def __call__(
        self,
        task_state: MWTTaskState,
        **kwargs,
    ) -> SolverResult:
        raise NotImplementedError


class CoTBrowserWebArenaSolver(BrowserWebArenaSolver):
    def __call__(
        self,
        task_state: MWTTaskState,
        **kwargs,
    ) -> SolverResult:
        base_prompt = COT_BROWSER_PROMPT["prompt"]
        current_example_template = COT_BROWSER_PROMPT["current_example"]
        current_example = current_example_template.format(
            objective=task_state.goal,
            observation=task_state.observation,
            url=task_state.url,
            previous_action=task_state.previous_action,
        )

        messages: OpenAICreateChatPrompt = [
            {"role": "system", "content": base_prompt},
            {"role": "user", "content": current_example},
        ]

        response = self.completion_fn(messages)
        parsed_action = self.extract_action(response.get_completions()[0])
        return SolverResult(parsed_action)

    def name(self) -> str:
        return "CoTBrowserWebArenaSolver"


class CoTBashBrowserWebArenaSolver(BrowserWebArenaSolver):
    def __call__(
        self,
        task_state: MWTTaskState,
        **kwargs,
    ) -> SolverResult:
        base_prompt = COT_BASH_BROWSER_PROMPT["prompt"]
        current_example_template = COT_BASH_BROWSER_PROMPT["current_example"]
        current_example = current_example_template.format(
            objective=task_state.goal,
            observation=task_state.observation,
            url=task_state.url if task_state.url else "None",
            previous_action=task_state.previous_action,
        )

        messages: OpenAICreateChatPrompt = [
            {"role": "system", "content": base_prompt},
            {"role": "user", "content": current_example},
        ]

        response = self.completion_fn(messages)
        parsed_action = self.extract_action(response.get_completions()[0])
        return SolverResult(parsed_action)

    def name(self) -> str:
        return "CoTBashBrowserWebArenaSolver"
