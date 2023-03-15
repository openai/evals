import logging
import math
import re

import evals
from evals.elsuite.basic.match import Match

logger = logging.getLogger(__name__)

NUMBER_RE = re.compile(r"-?\d+(\.\d+)?")


class ErfScoreMatch(Match):
    """
    If answers are distributed normally with mean m and standard deviation std, then a good way to score the accuracy of
    an answer would be to ask "how much probability density is there between the given answer and the correct answer?"

    Since the integral of the gaussian is math.erf, we're looking for abs(math.erf(given) - math.erf(expected)),
    where given and expected are scaled to have mean 0 and standard deviation 1 (i.e. (x - m) / std).

    Answer is expected to be given as a decimal number in data["sampled"].
    data["sampled"] may contain other space-separated tokens as long as they don't parse as decimal numbers.

    "ideal" is expected to contain a single decimal number (as a string).
    """

    def __init__(
        self,
        model_specs: evals.ModelSpecs,
        *args,
        mean: float,
        std: float,
        **kwargs,
    ):
        super().__init__(model_specs, *args, **kwargs)
        self.mean = mean
        self.std = std

    def run(self, recorder):
        samples = evals.get_jsonl(self.samples_jsonl)
        self.eval_all_samples(recorder, samples)
        events = recorder.get_events("match")
        return {
            "accuracy": self.calculate_accuracy(events),
        }

    def calculate_accuracy(self, events):
        return sum(
            self.calculate_erf_diff(self.parse(e.data["expected"]), self.parse(e.data["sampled"]))
            for e in events
        )

    def parse(self, s):
        numbers = [n for n in s.split() if NUMBER_RE.match(n)]
        if len(numbers) == 1:
            (number,) = numbers
            return float(number)
        logger.warn(f"Can't parse as number: {s}")
        return float("inf")

    def calculate_erf_diff(self, expected, picked):
        return abs(math.erf(self.scale(expected)) - math.erf(self.scale(picked)))

    def scale(self, x):
        return (x - self.mean) / self.std
