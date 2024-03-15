import re
from collections import Counter
from typing import Union

import numpy as np


def yellow_string(str: str) -> str:
    return f"\033[1;33m{str}\033[0m"


def median(numbers: list[int]) -> int:
    """
    Returns the median of the given list of numbers. If the list has an even
    number of elements, the arithmetic mean of the two middle elements of the
    sorted list is returned.
    """
    return np.median(numbers)


def mode(numbers: list[int]) -> int:
    """
    Returns the mode of the given list of numbers. If there are multiple modes,
    the largest mode is returned.
    """
    frequency = {}
    for number in numbers:
        frequency[number] = frequency.get(number, 0) + 1

    max_frequency = max(frequency.values())
    candidates = [number for number, freq in frequency.items() if freq == max_frequency]

    return max(candidates)


task_to_fn = {"median": median, "mode": mode}


def parse_solver_output(solver_output: str, task: str) -> Union[int, None]:
    solver_string = solver_output.strip().lower()
    pattern = rf"\[{task}: (\d+(?:\.\d+)?)\]"

    match = re.search(pattern, solver_string)

    if match:
        try:
            output = float(match.group(1))
        except ValueError:
            output = None
    else:
        output = None

    return output


def compute_mode_state(curr_list: list[int]) -> dict:
    counter = Counter(curr_list)
    return dict(counter)


def compute_median_state(curr_list: list[int]) -> dict:
    sorted_list = sorted(curr_list)
    return sorted_list


def compute_state(curr_list: list[int], task) -> dict:
    if task == "mode":
        return {
            "task_name": task,
            "state_label": "number to count",
            "state_data": compute_mode_state(curr_list),
        }
    else:
        return {
            "task_name": task,
            "state_label": "sorted list of shown numbers",
            "state_data": compute_median_state(curr_list),
        }
