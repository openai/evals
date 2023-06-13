from typing import Sequence

from pytest import mark
from evals.metrics import get_weighted_mean

@mark.parametrize(
    "values, weights, expected_weighted_mean",
    [
        ([1, 2, 3], [1, 1, 1], 2),
        ([1, 2, 3], [1, 2, 3], 2.3333333333333335),
        ([1, 2, 3], [3, 2, 1], 1.6666666666666667),
        ([1, 2, 3], [1, 0, 0], 1),
        ([1, 2, 3], [0, 1, 0], 2),
        ([1, 2, 3], [0, 0, 1], 3),
    ]
)
def test_get_weighted_mean(values: Sequence[float], weights: Sequence[float], expected_weighted_mean: float):
    assert get_weighted_mean(values, weights) == expected_weighted_mean
