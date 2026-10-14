"""
This file defines various common metrics of interest.
"""

import random
from typing import Optional, Sequence, Set

import numpy as np

from evals.record import Event


def get_accuracy(events: Sequence[Event]) -> float:
    num_correct = sum(int(event.data["correct"]) for event in events)
    num_total = len(events)
    if num_total == 0:
        return float("nan")
    else:
        return num_correct / num_total


def get_bootstrap_accuracy_std(events: Sequence[Event], num_samples: int = 1000) -> float:
    vals = [m.data["correct"] for m in events]
    return np.std([np.mean(random.sample(vals, len(vals) // 2)) for _ in range(num_samples)])


def get_confusion_matrix(
    matches: Sequence[Event], class_labels: Optional[Set] = None
) -> np.ndarray:
    labels = {match.data["expected"] for match in matches}
    if class_labels is None:
        labels = {label: i for i, label in enumerate(sorted(labels))}
    else:
        assert labels.issubset(class_labels)
        labels = {label: i for i, label in enumerate(class_labels)}
    result = np.zeros((len(labels), len(labels) + 1), dtype=int)
    for match in matches:
        i = labels[match.data["expected"]]
        j = labels.get(match.data["picked"], len(labels))
        result[i, j] += 1
    return result


def compute_matthew_corr(confusion_matrix: np.ndarray) -> float:
    assert confusion_matrix.shape == (2, 3), f"Got shape: {confusion_matrix.shape}"
    r = confusion_matrix[:, :2].copy()
    r[:, 0] += confusion_matrix[:, 2]
    denom = np.sqrt(r[1, :].sum() * r[0, :].sum() * r[:, 0].sum() * r[:, 1].sum())
    if denom == 0:
        return 0.0
    return float((r[1, 1] * r[0, 0] - r[1, 0] * r[0, 1]) / denom)


def compute_precision(confusion_matrix: np.ndarray, idx: int = 0) -> float:
    denom = confusion_matrix[:, idx].sum()
    if denom == 0:
        return 0.0
    return float(confusion_matrix[idx, idx] / denom)


def compute_recall(confusion_matrix: np.ndarray, idx: int = 0) -> float:
    denom = confusion_matrix[idx, :].sum()
    if denom == 0:
        return 0.0
    return float(confusion_matrix[idx, idx] / denom)


def compute_f_score(confusion_matrix: np.ndarray, idx: int = 0, beta: float = 1.0) -> float:
    precision = compute_precision(confusion_matrix, idx=idx)
    recall = compute_recall(confusion_matrix, idx=idx)
    denom = beta**2 * precision + recall
    if denom == 0:
        return 0.0
    return float((1 + beta**2) * (precision * recall) / denom)


def compute_averaged_f_score(
    confusion_matrix: np.ndarray, beta: float = 1.0, average: str = "macro"
) -> float:
    assert average in ["macro"]
    if confusion_matrix.shape[0] == 0:
        return 0.0
    f_scores = []
    for i in range(confusion_matrix.shape[0]):
        f_scores.append(compute_f_score(confusion_matrix, idx=i, beta=beta))
    return float(np.array(f_scores).mean())
