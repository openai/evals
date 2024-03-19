import ast
import logging
from typing import Sequence

logger = logging.getLogger(__name__)


def calculate_accuracy(tp: int, fp: int, tn: int, fn: int):
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return accuracy


def calculate_precision(tp: int, fp: int):
    if tp + fp == 0:
        return 0

    precision = tp / (tp + fp)
    return precision


def calculate_recall(tp: int, fn: int):
    if tp + fn == 0:
        return 0

    recall = tp / (tp + fn)
    return recall


def calculate_f1(precision: float, recall: float):
    if precision + recall == 0:
        return 0

    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def precision_recall_fscore(metrics: Sequence[dict]):
    """
    Calculates prediction metrics, where positive class is a tool being bugged. Handles edge cases
    where solver never predicted a certain class
    """

    def tool_is_buggy(metric):
        return len(metric["bugs"]) > 0

    # Calculate tp, fp, tn, fn
    tp = len([i for i in metrics if i["solver_predicted_bug"] and tool_is_buggy(i)])
    fn = len([i for i in metrics if not i["solver_predicted_bug"] and tool_is_buggy(i)])

    fp = len([i for i in metrics if i["solver_predicted_bug"] and not tool_is_buggy(i)])
    tn = len([i for i in metrics if not i["solver_predicted_bug"] and not tool_is_buggy(i)])

    # Calculate accuracy
    accuracy = calculate_accuracy(tp, fp, tn, fn)

    # If solver never predicts positive class, map each of the following to 0, not nan
    precision = calculate_precision(tp, fp)
    recall = calculate_recall(tp, fn)
    f1 = calculate_f1(precision, recall)

    return tp, fp, tn, fn, accuracy, precision, recall, f1


def try_cast_from_str(n: str, cast_type: type):
    """
    Given string n, cast to specified type and return. Warns and returns None
    if this fails
    """
    if cast_type not in (str, int, float, list):
        return None

    try:
        if cast_type == str:
            return str(n)
        elif cast_type == int:
            return int(n)
        elif cast_type == float:
            return float(n)
        elif cast_type == list:
            return ast.literal_eval(n)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None
