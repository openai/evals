import logging
import random
from typing import Callable, Dict, Optional, Union

# pip install https://github.com/amazon-science/mxeval if it doesnot exist
from mxeval.execution import check_correctness

import evals
import evals.metrics
from evals import completion_query
from evals.base import ModelSpec
from evals.prompt.base import OpenAICreateChatPrompt, OpenAICreatePrompt, Prompt
from evals.record import record_match, record_sampling

# https://github.com/amazon-science/mxeval


logger = logging.getLogger(__name__)


class MultiHumanEval(evals.Eval):
    def __init__(self, language, train_jsonl, test_jsonl, train_samples_per_prompt=1, **kwargs):
        super().__init__(**kwargs)
        self.language = language
        self.train_jsonl = train_jsonl
        self.test_jsonl = test_jsonl
        self.train_samples_per_prompt = train_samples_per_prompt

    def run(self, recorder):
        """
        Called by the `oaieval` CLI to run the eval. The `eval_all_samples` method calls `eval_sample`.
        """
        self.train_samples = evals.get_jsonl(self.train_jsonl)
        test_samples = evals.get_jsonl(self.test_jsonl)
        self.eval_all_samples(recorder, test_samples)

        return {
            "accuracy": evals.metrics.get_accuracy(recorder.get_events("match")),
        }

    def display_messages(self, messages):
        for message in messages:
            print(f"{message['role']}:\n{message['content']}\n")

    def construct_messages(
        self,
        turn_idx,
        problem,
        execution_result=None,
        previous_response=None,
        messages=None,
        fewshot_examples=[],
    ):
        if turn_idx == 0:
            prompt = problem["prompt"]
        else:
            prompt = (
                "Below is the error message\n-----------------------\n"
                + execution_result
                + f"\n-----------------------\nNow, please solve the following problem again\n\n"
                + problem["prompt"]
            )

        if messages is None:
            assert turn_idx == 0
            # first turn: construct messages
            # (1) high level instruction
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert coder in all programming languages. Please continue writing code based on each function signature without repeat the function signature. If you write any explanations in English sentences, please wrap them in comments in the correct format according to that language.",
                },
            ]
            # (2) examples
            for fs_example in fewshot_examples:
                messages.append({"role": "user", "content": fs_example["prompt"]})
                messages.append({"role": "assistant", "content": fs_example["canonical_solution"]})
            # (3) add the actual prompt
            messages.append({"role": "user", "content": prompt})
        else:
            # second turn: append user and assistant messages
            assert (
                previous_response is not None
            ), "For next turn, we require providing the previous assistant's response"
            messages.append({"role": "assistant", "content": previous_response})
            messages.append({"role": "user", "content": prompt})
        return messages

    def eval_sample(self, test_sample, rng: random.Random):
        """
        Called by the `eval_all_samples` method to evaluate a single sample.

        ARGS
        ====
        `test_sample`: a line from the JSONL test file
        `rng`: should be used for any randomness that is needed during evaluation

        This method does the following:
        1. Generate a prompt that contains the task statement, a few examples, and the test question.
        2. Check if the model generates the correct answer.
        """
        fewshot_examples = rng.sample(self.train_samples, self.train_samples_per_prompt)

        num_attempts = 3
        execution_result, messages, response, passed = None, None, None, False
        for turn_idx in range(num_attempts):
            messages = self.construct_messages(
                turn_idx=turn_idx,
                problem=test_sample,
                execution_result=execution_result,
                previous_response=response,
                messages=messages,
                fewshot_examples=fewshot_examples,
            )
            response, passed, execution_result = check_sampled_text_execute(
                self.model_spec,
                prompt=messages,
                problem=test_sample,
                language=self.language,
            )
            if passed or turn_idx == num_attempts - 1:
                # TODO - modify this class later to make it suitable for the metric
                record_match(
                    passed,
                    expected="",
                    picked=response,
                    sampled=response,
                    turn_idx=turn_idx,
                )
            if passed:
                break


def check_sampled_text_execute(
    model_spec: ModelSpec,
    prompt: Union[OpenAICreatePrompt, OpenAICreateChatPrompt, Prompt],
    problem: Dict,
    language: str,
    *,
    options: Optional[list[str]] = None,
    separator: Callable[[str], bool] = None,
) -> Optional[str]:
    """
    Generates a completion using the prompt, checks whether the completion is
        one of the expected completions, and then records the result.

    ARGS
    ====
    `model_spec`: See `completion_query`.
    `prompt`: See `completion_query`.
    `options`: The list of canonical options, defaults to `expected` if None.
        The completion will be converted to one of these options.
    `expected`: The desired completion or the list of desired completions.
    `separator`: A callable which check the character sampled after the option
        to see if it is a valid separator.

    RETURNS
    =======
    The option that was picked, i.e., matched the completion, or None.
    """

    result, actual_prompt, metadata = completion_query(
        prompt=prompt,
        temperature=0.0,
        model_spec=model_spec,
    )
    choice = result["choices"][0]

    sampled = choice["text"].strip() if model_spec.strip_completion else choice["text"]

    if "```" in sampled:
        sampled = sampled.split("```")[1]

    execute_result = check_correctness(problem, sampled, language=language, timeout=20.0)
    # passed (true or false), result (string from execution)

    result = {
        "prompt": actual_prompt,
        "sampled": sampled,
        "options": options,
        "picked": "",
    }
    match = execute_result["passed"]
    result["expected"] = ""
    result["match"] = match
    result["metadata"] = metadata
    result["system_message"] = execute_result["result"]
    record_sampling(**result)
    return sampled, execute_result["passed"], execute_result["result"]
