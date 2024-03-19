import math
import random
import string
from typing import Callable, Union

import numpy as np

from evals.elsuite.bugged_tools.tools import ToolResult, ToolTaskState


def explicit_error(tool_task_state: ToolTaskState, non_bugged_func: Callable) -> ToolResult:
    correct_out = non_bugged_func(tool_task_state)
    if correct_out is None:
        return None

    if isinstance(correct_out.output, int) or isinstance(correct_out.output, float):
        return ToolResult(math.nan)
    elif isinstance(correct_out.output, str):
        return ToolResult("ERROR")


def offset_numeric(correct_output: Union[int, float], offset: Union[int, float]) -> ToolResult:
    """
    Returns +-offset the correct output
    """
    offset = random.choice([-offset, offset])
    out = correct_output + offset  # Cast to type of correct output
    return ToolResult(out)


def small_offset(tool_task_state: ToolTaskState, non_bugged_func: Callable) -> ToolResult:
    def _small_offset_string(correct_output: str) -> ToolResult:
        """
        Changes a randomly selected character of the correct output to a random letter,
        digit or punctuaition
        """
        pos = random.randint(0, len(correct_output) - 1)
        random_char = random.choice(string.ascii_letters + string.digits + string.punctuation)
        out = correct_output[:pos] + random_char + correct_output[pos + 1 :]
        return ToolResult(out)

    correct_out = non_bugged_func(tool_task_state)
    if correct_out is None:
        return None

    if isinstance(correct_out.output, int) or isinstance(correct_out.output, float):
        return offset_numeric(correct_out.output, 1)
    elif isinstance(correct_out.output, str):
        return _small_offset_string(correct_out.output)


def large_offset(tool_task_state: ToolTaskState, non_bugged_func: Callable) -> ToolResult:
    def _large_offset_string(correct_output: str) -> ToolResult:
        """
        Changes several randomly selected characters of the correct output to a random
        letter, digit or punctuation
        """
        num_substitutions = math.ceil(len(correct_output) / 2)
        pos = np.random.choice(range(len(correct_output)), num_substitutions, replace=False)
        for idx in pos:
            random_char = random.choice(string.ascii_letters + string.digits + string.punctuation)
            correct_output = correct_output[:idx] + random_char + correct_output[idx + 1 :]
        return ToolResult(correct_output)

    correct_out = non_bugged_func(tool_task_state)
    if correct_out is None:
        return None

    if isinstance(correct_out.output, int) or isinstance(correct_out.output, float):
        return offset_numeric(correct_out.output, 10)
    elif isinstance(correct_out.output, str):
        return _large_offset_string(correct_out.output)


def random_output(tool_task_state: ToolTaskState, non_bugged_func: Callable) -> ToolResult:
    def _random_output_numeric(correct_output: Union[int, float]) -> ToolResult:
        """
        Returns random integer of same magnitude as correct answer
        """
        target_magnitude = len(str(int(correct_output))) - 1  # Cast to int to remove decimals
        lower_bound = 10**target_magnitude
        upper_bound = 10 ** (target_magnitude + 1) - 1

        out = correct_output
        while out == correct_output:
            out = random.randint(lower_bound, upper_bound)
            out *= random.choice([-1, 1])

        return ToolResult(out)

    def _random_output_string(correct_output: str) -> ToolResult:
        """
        Returns a random string of the same length as the correct answer
        """
        target_len = len(correct_output)
        out = correct_output
        while out == correct_output:
            out = "".join(
                random.choice(string.ascii_letters + string.digits) for _ in range(target_len)
            )
        return ToolResult(out)

    correct_out = non_bugged_func(tool_task_state)
    if correct_out is None:
        return None

    if isinstance(correct_out.output, int) or isinstance(correct_out.output, float):
        return _random_output_numeric(correct_out.output)
    elif isinstance(correct_out.output, str):
        return _random_output_string(correct_out.output)


def incorrect_type(tool_task_state: ToolTaskState, non_bugged_func: Callable) -> ToolResult:
    """
    Returns an output of the incorrect type
    """

    def _incorrect_type_numeric() -> ToolResult:
        words = [
            "import",
            "dog",
            "grape",
            "alice",
            "Sorry",
            "rain",
            "computer",
            "running",
            "bright",
        ]
        random_word = random.choice(words)
        return ToolResult(random_word)

    def _incorrect_type_string() -> ToolResult:
        num = random.choice(range(10))
        return ToolResult(num)

    correct_out = non_bugged_func(tool_task_state)
    if correct_out is None:
        return None

    if isinstance(correct_out.output, int) or isinstance(correct_out.output, float):
        return _incorrect_type_numeric()
    elif isinstance(correct_out.output, str):
        return _incorrect_type_string()


ALL_BUGS = {
    "explicit_error": explicit_error,
    "small_offset": small_offset,
    "large_offset": large_offset,
    "random_output": random_output,
    "incorrect_type": incorrect_type,
}


if __name__ == "__main__":
    from evals.elsuite.bugged_tools.tools import Double, ReverseStr
    from evals.task_state import Message

    x = "abcd"
    example_task_state = ToolTaskState(
        task_description="", messages=[Message(role="user", content=x)], current_state=None
    )
    print(
        f"Small offset for {ReverseStr} on input {x}: {small_offset(example_task_state, ReverseStr())}"
    )
    print(
        f"Large offset for {ReverseStr} on input {x}: {large_offset(example_task_state, ReverseStr())}"
    )
    print(
        f"Random output for {ReverseStr} on input {x}: {random_output(example_task_state, ReverseStr())}"
    )
    print(
        f"Incorrect type for {ReverseStr} on input {x}: {incorrect_type(example_task_state, ReverseStr())}"
    )

    x = "15"
    example_task_state = ToolTaskState(
        task_description="", messages=[Message(role="user", content=x)], current_state=None
    )
    print(f"Small offset for {Double} on input {x}: {small_offset(example_task_state, Double())}")
    print(f"Large offset for {Double} on input {x}: {large_offset(example_task_state, Double())}")
    print(f"Random output for {Double} on input {x}: {random_output(example_task_state, Double())}")
    print(
        f"Incorrect type for {Double} on input {x}: {incorrect_type(example_task_state, Double())}"
    )
