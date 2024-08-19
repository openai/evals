import abc
import random

import numpy as np

from evals.elsuite.identifying_variables.structs import Sample


class RendererBase(abc.ABC):
    def __init__(self, rng: random.Random, np_rng: np.random.Generator) -> None:
        self.rng = rng
        self.np_rng = np_rng

    @abc.abstractmethod
    def render_obs(self, sample: Sample) -> str:
        raise NotImplementedError
