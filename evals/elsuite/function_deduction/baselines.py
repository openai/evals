import logging
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import entropy

from evals.data import get_jsonl
from evals.elsuite.function_deduction.eval import CurrentState, Sample
from evals.registry import Registry
from evals.solvers.solver import Solver, SolverResult
from evals.task_state import TaskState


class AverageBaseline(Solver):
    """
    For given test inputs (x, y, z):
    * Ask about values of (x-1, x+1, y-1, y+1, z-1, z+1)
    * Make three guesses: round/floor/ceil of average values for neighboring numbers
    If didn't succeed in 9 rounds (6x ask 3x guess) - surrender.

    Note: This algorithm fails on the edge cases where, for any of the inputs i:
    - i-1 or i+1 is out of range
    - i-1 or i+1 are part of the test inputs
    In this scenario, the algorithm will fail at the _get_guess stage and skip the guessing.
    """

    def __init__(self, registry=None):
        pass

    def _solve(self, task_state: TaskState):
        cs: CurrentState = task_state.current_state

        assert len(cs.test_inputs) == 3, "AverageBaseline assumes 3 test inputs"

        if cs.round_ix < 6:
            response = self._get_ask(cs.test_inputs, cs.round_ix)
        elif 6 <= cs.round_ix < 9:
            response = self._get_guess(cs.test_inputs, cs.known_values, cs.round_ix - 6)
        else:
            response = "I've run out of ideas sorry :("
        return SolverResult(response)

    def _get_guess(self, test_inputs, known_values: dict[int, int], guess_round_ix) -> str:
        known_values = {
            x: y for x, y in known_values.items() if x - 1 in test_inputs or x + 1 in test_inputs
        }

        pairs = [[], [], []]
        for i, test_input in enumerate(test_inputs):
            try:
                lower = known_values[test_input - 1]
                higher = known_values[test_input + 1]
            except KeyError:
                return "Unfortunately I don't have enough data to make a guess, will pass."
            pairs[i] = [lower, higher]

        funcs = [round, math.floor, math.ceil]
        func = funcs[guess_round_ix]
        vals = [func((pair[0] + pair[1]) / 2) for pair in pairs]
        return " ".join([str(x) for x in vals])

    def _get_ask(self, test_inputs, round_ix) -> str:
        queries = []
        for x in test_inputs:
            queries.append(x - 1)
            queries.append(x + 1)

        ask = queries[round_ix]
        if ask in test_inputs or ask < 0 or ask > 100:
            logging.warning(
                f"Invalid query on inputs {test_inputs}: {ask}. AverageBaseline algorithm will fail."
            )
        return str(ask)


class FullKnowledge(Solver):
    """Assuming solver knows all the samples, how well would it perform?

    Two modes - "random", where it selects random integer when asking,
    and "best" where it selects the best integer.

    The "best" mode should be close to unbeatable (except for lucky guesses).
    """

    def __init__(self, mode: str, samples_jsonl: str, registry: Registry):
        assert mode in ("random", "best"), "mode must be either random or best"
        self.mode = mode
        self._all_samples = self._get_samples(samples_jsonl, registry._registry_paths[0])
        self._rng = np.random.default_rng()

    def _solve(self, task_state: TaskState):
        cs: CurrentState = task_state.current_state

        matching_samples = self._get_matching_samples(cs.known_values)
        if len(matching_samples) > 1:
            if self.mode == "random":
                response = self._get_ask_random(cs.known_values)
            else:
                response = self._get_ask_best(matching_samples)
        else:
            sample_values = matching_samples[0].values
            result = [sample_values[test_input] for test_input in cs.test_inputs]
            response = " ".join([str(x) for x in result])
        return SolverResult(str(response))

    def _get_matching_samples(self, known_values):
        def matches(sample: Sample) -> bool:
            for key, val in known_values.items():
                if sample.values[key] != val:
                    return False
            return True

        return [sample for sample in self._all_samples if matches(sample)]

    def _get_ask_best(self, samples):
        def get_entropy(x: int) -> float:
            values = [sample.values[x] for sample in samples]
            counter = Counter(values)
            return entropy([val for val in counter.values()])

        return max(range(0, 101), key=get_entropy)

    def _get_ask_random(self, known_values):
        while True:
            x = self._rng.integers(0, 100)
            if x not in known_values:
                return x

    def _get_samples(self, samples_jsonl: str, registry_path: Path):
        path = registry_path / "data" / samples_jsonl
        return [Sample(**x) for x in get_jsonl(path.as_posix())]
