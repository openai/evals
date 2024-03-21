"""Latent functions for the project."""
import numpy as np


def linear(x: np.ndarray, grad: float, bias: float) -> np.ndarray:
    return grad * x + bias


def quadratic(x: np.ndarray, grad: float, bias: float) -> np.ndarray:
    return grad * x**2 + bias


def random_uniform(num_samples, min_v, max_v, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(min_v, max_v, num_samples)


def random_ints(num_samples, min_v, max_v, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(min_v, max_v, num_samples)


LATENT_FUNC_MAP = {
    "linear": linear,
    "quadratic": quadratic,
}
LATENT_FUNC_KWARG_MAP = {
    "linear": {
        "grad": {"min_v": -10, "max_v": 10},
        "bias": {"min_v": -100, "max_v": 100},
    },
    "quadratic": {
        "grad": {"min_v": -10, "max_v": 10},
        "bias": {"min_v": -100, "max_v": 100},
    },
}

DISTRIBUTIONS = {
    # "random_uniform": random_uniform,
    "random_ints": random_ints,
}
DISTRIBUTIONS_KWARG_MAP = {
    "random_uniform": {"min_v": -1, "max_v": 1},
    "random_ints": {"min_v": -100, "max_v": 100},
}
