from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest

from evals import metrics


@pytest.mark.parametrize(
    "event_labels, expected",
    [
        ([True, True], 1.0),
        ([True, False, False], 0.333),
        ([False, False], 0.0),
        ([], np.nan),
    ],
)
def test_get_accuracy(
    event_labels: List[bool],
    expected: float,
) -> None:
    events = [MagicMock(data={"correct": value}) for value in event_labels]
    np.testing.assert_allclose(expected, metrics.get_accuracy(events), rtol=1e-3)


@pytest.mark.parametrize(
    "event_labels, expected",
    [
        # A single sample must not produce NaN: every bootstrap resample equals
        # the one value, so the std of the resampled means is exactly 0.0.
        ([True], 0.0),
        # Constant inputs of any size have zero variance across resamples.
        ([True, True, True], 0.0),
        ([False, False], 0.0),
        # No events -> NaN, consistent with get_accuracy.
        ([], np.nan),
    ],
)
def test_get_bootstrap_accuracy_std_edge_cases(
    event_labels: List[bool],
    expected: float,
) -> None:
    events = [MagicMock(data={"correct": value}) for value in event_labels]
    np.testing.assert_allclose(
        expected, metrics.get_bootstrap_accuracy_std(events), rtol=1e-3
    )


def test_get_bootstrap_accuracy_std_is_finite_and_nonnegative() -> None:
    # A mixed set should yield a finite, non-negative std (never NaN).
    events = [MagicMock(data={"correct": v}) for v in [True, False, True, False]]
    std = metrics.get_bootstrap_accuracy_std(events, num_samples=200)
    assert np.isfinite(std)
    assert std >= 0.0
