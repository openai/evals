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


def test_compute_precision_recall_f_score_standard() -> None:
    # 2 classes: class 0 has 8 TP, 2 FP from class 1; class 1 has 7 TP, 1 FP from class 0
    # abstain column is 0
    cm = np.array(
        [
            [8, 1, 0],
            [2, 7, 0],
        ]
    )
    assert metrics.compute_precision(cm, idx=0) == 8 / 10  # 0.8
    assert metrics.compute_recall(cm, idx=0) == 8 / 9
    assert metrics.compute_precision(cm, idx=1) == 7 / 8  # 0.875
    assert metrics.compute_recall(cm, idx=1) == 7 / 9

    expected_f0 = 2 * (0.8 * (8 / 9)) / (0.8 + (8 / 9))
    assert np.isclose(metrics.compute_f_score(cm, idx=0), expected_f0)


def test_compute_precision_recall_zero_division() -> None:
    # Class 1 was never predicted (column 1 sum == 0)
    # Class 2 has 0 actual instances (row 2 sum == 0)
    cm = np.array(
        [
            [5, 0, 0, 0],
            [3, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    assert metrics.compute_precision(cm, idx=1) == 0.0
    assert metrics.compute_recall(cm, idx=2) == 0.0


def test_compute_f_score_zero_division() -> None:
    # Both precision and recall are 0
    cm = np.array(
        [
            [0, 5, 0],
            [5, 0, 0],
        ]
    )
    assert metrics.compute_precision(cm, idx=0) == 0.0
    assert metrics.compute_recall(cm, idx=0) == 0.0
    assert metrics.compute_f_score(cm, idx=0) == 0.0


def test_compute_averaged_f_score() -> None:
    cm = np.array(
        [
            [10, 0, 0],
            [5, 0, 0],
        ]
    )
    # Class 0: prec = 10/15 = 2/3, rec = 10/10 = 1.0 -> f1 = 2 * (2/3) / (5/3) = 0.8
    # Class 1: prec = 0.0, rec = 0.0 -> f1 = 0.0
    # Macro F1 should be (0.8 + 0.0) / 2 = 0.4 without producing NaN
    macro_f1 = metrics.compute_averaged_f_score(cm)
    assert not np.isnan(macro_f1)
    assert np.isclose(macro_f1, 0.4)

    # Empty confusion matrix
    empty_cm = np.zeros((0, 3), dtype=int)
    assert metrics.compute_averaged_f_score(empty_cm) == 0.0


def test_compute_matthew_corr_does_not_mutate_and_handles_zero_denom() -> None:
    cm = np.array([[8, 1, 1], [2, 7, 1]])
    cm_copy = cm.copy()
    mcc = metrics.compute_matthew_corr(cm)
    assert np.array_equal(cm, cm_copy)  # Input matrix must not be mutated
    assert not np.isnan(mcc)

    # All-zero confusion matrix handles division by zero safely
    zero_cm = np.zeros((2, 3), dtype=int)
    assert metrics.compute_matthew_corr(zero_cm) == 0.0
