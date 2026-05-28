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


def test_compute_matthew_corr_does_not_mutate_input() -> None:
    # compute_matthew_corr folds the abstain column into column 0; this must
    # happen on a copy so the caller's confusion matrix is left intact and any
    # other metric computed from it stays correct.
    confusion_matrix = np.array([[8, 1, 1], [2, 7, 1]], dtype=int)
    original = confusion_matrix.copy()

    mcc = metrics.compute_matthew_corr(confusion_matrix)

    np.testing.assert_allclose(mcc, 0.6124, rtol=1e-3)
    np.testing.assert_array_equal(confusion_matrix, original)
