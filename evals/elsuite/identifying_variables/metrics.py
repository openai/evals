from typing import Dict, List, Set

import numpy as np

from evals.elsuite.identifying_variables.utils import parse_solver_preds
from evals.solvers.solver import SolverResult


def compute_DCG(ranking: List[float], ceil_negs: bool = False) -> float:
    """
    Computes the DCG of a ranking
    """
    dcg = 0
    for i, rel in enumerate(ranking, start=1):
        if ceil_negs:
            rel = max(rel, 0)
        dcg += rel / np.log2(i + 1)  # (i+1) to avoid log_2(1) which = 0
    return dcg


def compute_nDCG(ranking: List[float], best: List[float], worst: List[float]) -> float:
    """
    Computes nDCG, allowing for negative scores, based on the nDCG variant
    from Gienapp et al. (2020) (https://dl.acm.org/doi/10.1145/3340531.3412123)
    """
    idcg = compute_DCG(best)
    min_dcg = compute_DCG(worst)
    dcg = compute_DCG(ranking)
    return (dcg - min_dcg) / (idcg - min_dcg)


def compute_metric_posthoc(
    metric: str, metric_entries: List[Dict], sampling_entries: List[Dict]
) -> float:
    """
    Computes a metric that was not logged by the eval, post-hoc, i.e.
    after the eval has run, by reading the log file.
    """
    metric_to_func = {
        "ctrl_recall": compute_ctrl_recall_posthoc,
    }
    if metric not in metric_to_func.keys():
        raise ValueError(f"Metric {metric} not supported")
    return metric_to_func[metric](metric_entries, sampling_entries)


def compute_ctrl_recall_posthoc(metric_entries: List[Dict], sampling_entries: List[Dict]) -> float:
    """
    Computes the average recall for identified control variables

    i.e. the no. of correctly identified control variables / no. gold control variables
    Averaged across the samples.

    - We skip any samples where the gold hypothesis is invalid
    - And we skip any samples where there are no control variables in the gold label,
    since recall is undefined in this case
    """
    recalls = []
    for metric_entry, sampling_entry in zip(metric_entries, sampling_entries):
        try:
            preds = parse_solver_preds(SolverResult(output=sampling_entry["sampled"][0]))
        except ValueError:  # in case of invalid solver output (violation)
            preds = None

        if metric_entry["gold_answer"]["valid_hypothesis"]:
            if preds and preds.ctrl_vars is not None:
                recall = compute_recall(
                    set(preds.ctrl_vars), set(metric_entry["gold_answer"]["ctrl_vars"])
                )
            else:
                # worst case scenario in case of violation or incorrect hyp validation
                recall = 0
        else:
            recall = np.nan
        recalls.append(recall)
    return np.nanmean(recalls).astype(float)


def compute_fallout(retrieved: Set[str], gold_relevants: Set[str], num_irrelevant: int) -> float:
    """
    Computes fallout for a sample
    Number of retrieved irrelevant items / number of irrelevant items

    This value is undefined when there are no irrelevant items in the gold label
    """
    if num_irrelevant == 0:
        # undefined
        return np.nan
    retrieved_irrel_count = len([r for r in retrieved if r not in gold_relevants])
    return retrieved_irrel_count / num_irrelevant


def compute_recall(retrieved: Set[str], gold_relevants: Set[str]):
    """
    Computes recall for a sample
    Number of retrieved relevant items / number of relevant items

    This value is undefined when there are no relevant items in the gold label
    """
    num_relevant = len(gold_relevants)
    if num_relevant == 0:
        # undefined
        return np.nan
    retrieved_rel_count = len([r for r in retrieved if r in gold_relevants])
    return retrieved_rel_count / num_relevant
