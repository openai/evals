import os
from abc import abstractmethod
from typing import Any, Dict, Sequence

from evals.completion_fns.openai import OpenAIChatCompletionFn
from evals.elsuite.sandbagging.defaults import (
    SANDBAGGING_IMPROVED_GPT_4,
    SANDBAGGING_IMPROVED_GPT_35,
    SANDBAGGING_SIMPLE,
    SANDBAGGING_SUBSET_SIMPLE,
    SCRATCHPAD_MSG,
    SCRATCHPAD_QUESTION_MSG,
)
from evals.solvers.solver import Solver, SolverResult
from evals.task_state import TaskState

# HACK since we can't specify exact prompt in .yaml
all_prompts = {
    "benign": SANDBAGGING_SIMPLE,
    "subset": SANDBAGGING_SUBSET_SIMPLE,
    "gpt-4": SANDBAGGING_IMPROVED_GPT_4,
    "gpt-3.5": SANDBAGGING_IMPROVED_GPT_35,
}


class StrategySolver(Solver):
    """
    Solver for MMLU, adapts its strategy based off the parameters
    passed at initialization. Generalized for both LMs and humans
    """

    def __init__(
        self,
        model_name: str,
        sandbag_prompt: str,
        scratchpad: bool,
        no_available_prev_ans: int,
        **kwargs,
    ):
        self.model_name = model_name
        self.sandbag_prompt = sandbag_prompt
        self.scratchpad = scratchpad
        self.no_available_prev_ans = no_available_prev_ans

        self.prev_ans = []

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def _predict_answer(self, prompt: Sequence[Dict], **kwargs) -> str:
        pass

    @abstractmethod
    def _produce_scratchpad(self, prompt: Sequence[Dict]):
        """
        Prompt model to produce private scratchpad. We always use temp=1
        for the scratchpad when possible
        """

    def modify_target(self, target: int):
        """
        In sandbagging.py, we instantiate different MMLU evals to sandbag one, each with
        a different target accuracy. Since they all use the same solver object, we must
        modify the object repeatedly to communicate the new target
        """
        self.target = target

    def _add_ans_to_memory(self, prompt: Sequence[Dict], response: str):
        """
        Add the most recent question and models answer to memory. If we're using a
        scratchpad, then add the relevant system messages and models private messages
        """
        if self.scratchpad:
            # Add (question, scratch system prompt, scratch, question system propmt)
            self.prev_ans += prompt[-4:]
        else:
            self.prev_ans += prompt[-1:]  # Add (question)
        self.prev_ans += [{"role": "assistant", "content": response}]

    def _construct_prompt(self, task_state: TaskState) -> Sequence[Dict]:
        """
        Construct prompt using task state, depending on the strategy
        specified on init
        """
        prompt = []
        task_description = task_state.task_description

        # Add additional sandbagging prompt
        if self.sandbag_prompt != "None":
            p = all_prompts[self.sandbag_prompt]
            target_prompt = p.substitute(
                model_name=self.model_name,
                target_acc=self.target,
            )
            prompt += [
                {"role": "system", "content": target_prompt},
            ]

        # Add solvers previous answers
        if self.no_available_prev_ans > 0:
            if self.sandbag_prompt:
                # Get all tuples (question, scratch system prompt, scratch, question system propmt, answer)
                prev_n_ans = self.prev_ans[-self.no_available_prev_ans * 5 :]
            else:
                # Get all tuples (question, answer)
                prev_n_ans = self.prev_ans[-self.no_available_prev_ans * 2 :]
        else:
            prev_n_ans = []

        # Put all together to make prompt
        prompt += (
            [
                {"role": "system", "content": task_description},
            ]
            + prev_n_ans
            + [msg.to_dict() for msg in task_state.messages]
        )

        # If we're using scratchpad, first prompt model for reasoning, then include this in the
        # final prompt
        if self.scratchpad:
            prompt = self._produce_scratchpad(prompt)
            prompt += [{"role": "system", "content": SCRATCHPAD_QUESTION_MSG}]

        return prompt

    def __call__(self, task_state: TaskState, **kwargs) -> (Sequence[Dict], SolverResult):
        prompt = self._construct_prompt(task_state)
        result = self._predict_answer(prompt, **kwargs)

        self._add_ans_to_memory(prompt, result)
        return SolverResult(
            result,
            prompt=prompt,
        )


class LMSolver(StrategySolver):
    def __init__(
        self,
        completion_fn_options: Dict[str, Any],
        model_name: str,
        sandbag_prompt: str,
        scratchpad: bool,
        no_prev_ans: int,
        temperature: float = 0.0,
        **kwargs,
    ):
        print(scratchpad is True)
        super().__init__(
            model_name=model_name,
            sandbag_prompt=sandbag_prompt,
            scratchpad=scratchpad,
            no_available_prev_ans=no_prev_ans,
            **kwargs,
        )

        completion_fn_options["extra_options"]["temperature"] = temperature

        self.completion_fn = OpenAIChatCompletionFn(**completion_fn_options)
        if scratchpad:
            completion_fn_options["extra_options"]["temperature"] = 1.0
            self.scratchpad_completion_fn = OpenAIChatCompletionFn(**completion_fn_options)

    def name(self) -> str:
        return "LM-Solver"

    def _produce_scratchpad(self, prompt: Sequence[Dict]):
        prompt += [{"role": "system", "content": SCRATCHPAD_MSG}]
        result = self._predict_answer(prompt)
        prompt += [{"role": "assistant", "content": result}]
        return prompt

    def _predict_answer(self, prompt: Sequence[Dict], **kwargs) -> str:
        if self.scratchpad:
            fn = self.scratchpad_completion_fn
        else:
            fn = self.completion_fn
        completion_result = fn(prompt=prompt, **kwargs)
        return completion_result.get_completions()[0]


class CLISolver(StrategySolver):
    """
    Human solver which uses the CLI. This only makes sense with EVALS_SEQUENTIAL=1
    """

    def __init__(
        self,
        scratchpad: bool,
        no_prev_ans: int,
        **kwargs,
    ):
        assert (
            int(os.environ.get("EVALS_SEQUENTIAL", "-1")) == 1
        ), f"EVALS_SEQUENTIAL should be 1 for CLI solver but is {os.environ.get('EVALS_SEQUENTIAL', 'not specified')}"

        super().__init__(
            model_name="a human",
            sandbag_prompt="None",
            scratchpad=scratchpad,
            no_available_prev_ans=no_prev_ans,
            **kwargs,
        )

    def name(self) -> str:
        return "CLI-Solver"

    def _produce_scratchpad(self, prompt: Sequence[Dict]):
        prompt += [{"role": "system", "content": SCRATCHPAD_MSG}]
        result = self._predict_answer(prompt)
        prompt += [{"role": "assistant", "content": result}]
        return prompt

    def _predict_answer(self, prompt: Sequence[Dict], **kwargs) -> str:
        return input(prompt)
