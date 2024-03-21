"""
Code for generating .jsonl dataset for identifying variables eval

Use default argparse args to replicate the dataset used for the report
"""

from dataclasses import asdict
import os
import argparse
from typing import Dict, List, Optional, Set, Tuple, Any
import json
import copy

from tqdm.auto import tqdm
import networkx as nx
import numpy as np

import evals.elsuite.identifying_variables.latent_funcs as latent_funcs
from evals.elsuite.identifying_variables.graph_utils import (
    gen_random_forest,
    gen_random_forest_tree_size,
    find_graph_roots,
    find_unconnected_nodes_pair,
    find_connected_nodes_pair,
)
from evals.elsuite.identifying_variables.utils import sample_serializer
from evals.elsuite.identifying_variables.structs import Sample, Answer
import evals.elsuite.identifying_variables.constants as constants


def write_to_jsonl(
    samples: List[Sample],
    jsonl_path: str,
):
    with open(jsonl_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(asdict(sample), default=sample_serializer) + "\n")


def random_latent_func_meta(
    np_rng: np.random.Generator, input_x: Optional[str] = None
) -> Dict:
    """
    Generates random metadata for defining a latent function

    Args:
        input_x (Optional[str]): Name of input variable. If None, then
        the latent function is a distribution, not dependent on any input.
    """
    if input_x is None:
        latent_func_name = np_rng.choice(list(latent_funcs.DISTRIBUTIONS.keys()))
        predefined_kwargs = latent_funcs.DISTRIBUTIONS_KWARG_MAP[latent_func_name]
        kwargs = {**predefined_kwargs}
        return {"name": latent_func_name, "kwargs": kwargs}
    else:
        latent_func_name = np_rng.choice(list(latent_funcs.LATENT_FUNC_MAP.keys()))
        predefined_kwargs = latent_funcs.LATENT_FUNC_KWARG_MAP[latent_func_name]
        kwargs = {}
        for kwarg, min_max in predefined_kwargs.items():
            kwarg_value = np_rng.integers(min_max["min_v"], min_max["max_v"])
            while kwarg == "grad" and kwarg_value == 0:
                # dont allow 0 gradient
                kwarg_value = np_rng.integers(min_max["min_v"], min_max["max_v"])
            kwargs[kwarg] = kwarg_value
        return {"name": latent_func_name, "input_x": input_x, "kwargs": kwargs}


def build_var_metadata(
    causal_graph: nx.DiGraph,
    sparse_var_rate: float,
    np_rng: np.random.Generator,
) -> Dict:
    """
    Builds the variable metadata for a sample, containing
    information on how each variable is generated and which variables
    it is correlated with.

    Args:
        causal_graph (nx.DiGraph): Causal graph of the sample.
        sparse_var_rate (float): Percentage of variables that should be sparsified.
        max_sparsity (float): Maximum sparsity rate for sparse variables.
        np_rng (np.random.Generator): Random number generator to be used.
    """
    var_metadata = {}

    roots = find_graph_roots(causal_graph)
    root_to_descendants = {r: nx.descendants(causal_graph, r) for r in roots}
    node_to_root = {
        n: root
        for root, descendants in root_to_descendants.items()
        for n in descendants
    }

    for var in causal_graph:
        if var in roots:
            latent_func_meta = random_latent_func_meta(np_rng, input_x=None)
            var_root = var
        else:
            parent = next(causal_graph.predecessors(var))
            latent_func_meta = random_latent_func_meta(np_rng, input_x=parent)
            var_root = node_to_root[var]
        # variables with a common root are correlated. Need to copy to avoid mutation
        corrs: Set[str] = set(root_to_descendants[var_root])
        if var_root != var:
            # remove self-correlation, add correlation to root itself
            corrs.remove(var)
            corrs.add(var_root)

        var_metadata[var] = {
            "gen_method": latent_func_meta,
            "corrs": corrs,
            "extra": {"sparsity_rate": 0},
        }

    # add sparsity
    var_metadata = sparsify_data(var_metadata, sparse_var_rate, np_rng)

    return var_metadata


def sparsify_data(var_metadata, sparse_var_rate, np_rng):
    num_observed_vars = 0
    orig_var_metadata = copy.deepcopy(var_metadata)
    for var in var_metadata.keys():
        if np_rng.uniform(0, 1) < sparse_var_rate:
            sparsity_rate = np_rng.uniform(
                low=constants.MIN_SPARSITY, high=constants.MAX_SPARSITY
            )
            var_metadata[var]["extra"]["sparsity_rate"] = sparsity_rate
            if sparsity_rate > constants.SPARSITY_FOR_UNOBS:
                # remove unobserved variables from correlations
                for corr_var in var_metadata[var]["corrs"]:
                    var_metadata[corr_var]["corrs"].remove(var)
                var_metadata[var]["corrs"] = set()
            else:
                num_observed_vars += 1
        else:
            num_observed_vars += 1

    # if less than 2 observed variables, sparsification was too much, try again
    if num_observed_vars < 2:
        var_metadata = sparsify_data(orig_var_metadata, sparse_var_rate, np_rng)

    return var_metadata


def gen_sample_balanced_ctrl_vars(
    signal_noise_ratio: Optional[float], np_rng: np.random.Generator
) -> Sample:
    """
    Generates a sample for the dataset, containing information on how a set
    of variables are interlinked, and which hypotheses are currently held.

    This differs from gen_sample in the following ways:

    To simplify:
     - The total number of variables in a given sample is fixed to MAX_VARS
     - The hypothesis is always valid

    The number of control variables is sampled uniformly between 0 and MAX_VARS-2
    (we subtract 2 since two variables are involved in the hypothesis)
    """
    sample_metadata = {"snr": signal_noise_ratio}

    n_vars = constants.MAX_VARS

    sparse_var_rate = np_rng.uniform(
        low=constants.MIN_SPARSE_VAR_RATE, high=constants.MAX_SPARSE_VAR_RATE
    )  # perc of variables to sparsify

    var_ids = np_rng.choice(np.arange(1000, 10000), size=n_vars, replace=False).astype(
        str
    )
    var_names = [f"x_{var_id}" for var_id in var_ids]

    num_ctrl_vars = np_rng.integers(low=0, high=n_vars - 1)  # high is exclusive

    causal_graph = gen_random_forest_tree_size(
        nodes=var_names, tree_size=num_ctrl_vars + 2, np_rng=np_rng
    )

    variable_metadata = build_var_metadata(causal_graph, sparse_var_rate, np_rng)

    target_hypothesis = find_connected_nodes_pair(causal_graph, np_rng)
    target_hyp_is_valid = (
        parse_target_hyp(target_hypothesis, variable_metadata)[0]
        if target_hypothesis is not None
        else None
    )
    # try again if the sparsification caused the hypothesis to be invalid
    if target_hypothesis is None or not target_hyp_is_valid:
        return gen_sample_balanced_ctrl_vars(signal_noise_ratio, np_rng)

    n_hypotheses = np_rng.integers(
        low=constants.MIN_HYPS,
        high=min(constants.MAX_HYPS, n_vars - 1) + 1,
    )
    hypotheses = gen_random_forest(var_names, total_edges=n_hypotheses, np_rng=np_rng)

    hypotheses = integrate_target_hyp(target_hypothesis, hypotheses, np_rng)

    gold_label, num_not_ctrl = determine_gold_label(
        target_hypothesis, variable_metadata, hypotheses
    )

    return Sample(
        variable_metadata=variable_metadata,
        hypotheses=hypotheses,
        target_hypothesis=target_hypothesis,
        sample_metadata=sample_metadata,
        # keep track of underlying ground truth in case want more in depth analysis
        causal_graph=causal_graph,
        gold_label=gold_label,
        num_not_ctrl=num_not_ctrl,
    )


def gen_sample(
    signal_noise_ratio: Optional[float],
    np_rng: np.random.Generator,
    valid_hyp_requested: Optional[bool] = None,
) -> Sample:
    """
    Generates a sample for the dataset, containing information on how a set
    of variables are interlinked, and which hypotheses are currently held.

    Args:
        signal_noise_ratio (float): Signal-to-noise ratio to be applied to the
            observations. If None, no noise is applied.
        np_rng (np.random.Generator): Random number generator to be used.
        valid_hyp_requested (Optional[bool]): Whether the target hypothesis should be
            valid. If None, will be randomly chosen.

    Returns:
        Sample: A sample as defined by the `Sample` dataclass.
    """
    sample_metadata = {"snr": signal_noise_ratio}

    n_vars = np_rng.integers(low=constants.MIN_VARS, high=constants.MAX_VARS + 1)
    sparse_var_rate = np_rng.uniform(
        low=constants.MIN_SPARSE_VAR_RATE, high=constants.MAX_SPARSE_VAR_RATE
    )  # perc of variables to sparsify

    var_ids = np_rng.choice(np.arange(1000, 10000), size=n_vars, replace=False).astype(
        str
    )
    var_names = [f"x_{var_id}" for var_id in var_ids]

    causal_graph = gen_random_forest(var_names, np_rng=np_rng)

    variable_metadata = build_var_metadata(causal_graph, sparse_var_rate, np_rng)

    n_hypotheses = np_rng.integers(
        low=constants.MIN_HYPS,
        high=min(constants.MAX_HYPS, n_vars - 1) + 1,
    )
    hypotheses = gen_random_forest(var_names, total_edges=n_hypotheses, np_rng=np_rng)

    if valid_hyp_requested is None:
        # 0.5 chance of valid hypothesis
        valid_hyp_requested = np_rng.uniform(0, 1) < 0.5

    if valid_hyp_requested:
        target_hypothesis = find_connected_nodes_pair(causal_graph, np_rng)
    else:
        target_hypothesis = find_unconnected_nodes_pair(causal_graph)

    target_hyp_is_valid = (
        parse_target_hyp(target_hypothesis, variable_metadata)[0]
        if target_hypothesis is not None
        else None
    )
    if target_hypothesis is None or target_hyp_is_valid != valid_hyp_requested:
        return gen_sample(signal_noise_ratio, np_rng, valid_hyp_requested)

    hypotheses = integrate_target_hyp(target_hypothesis, hypotheses, np_rng)

    gold_label, num_not_ctrl = determine_gold_label(
        target_hypothesis, variable_metadata, hypotheses
    )

    return Sample(
        variable_metadata=variable_metadata,
        hypotheses=hypotheses,
        target_hypothesis=target_hypothesis,
        sample_metadata=sample_metadata,
        # keep track of underlying ground truth in case want more in depth analysis
        causal_graph=causal_graph,
        gold_label=gold_label,
        num_not_ctrl=num_not_ctrl,
    )


def determine_gold_label(
    target_hyp, variable_metadata, hypotheses
) -> Tuple[Answer, Optional[int]]:
    """
    Determines the ideal `Answer` for a given sample. Additionally returns
    the number of variables not controlled for, if the hypothesis is valid,
    necessary for nDCG calculation.
    """
    valid_hypothesis, ind_var, dep_var = parse_target_hyp(target_hyp, variable_metadata)
    if not valid_hypothesis:
        ctrl_vars, not_ctrls = None, None
        num_not_ctrl = None
    else:
        ctrl_vars, not_ctrls = determine_ctrl_vars(
            variable_metadata, ind_var, dep_var, hypotheses
        )
        # worst case ctrl: all vars that aren't meant to be ctrld are ctrld
        num_not_ctrl = len(not_ctrls)

    return (
        Answer(
            valid_hypothesis=valid_hypothesis,
            ind_var=ind_var,
            dep_var=dep_var,
            ctrl_vars=ctrl_vars,
        ),
        num_not_ctrl,
    )


def parse_target_hyp(
    target_hyp: Tuple[str, str], variable_metadata: Dict[str, Any]
) -> Tuple[bool, Optional[str], Optional[str]]:
    """Implements decision tree in Figure 2 from eval spec"""
    proposed_ind = target_hyp[0]
    proposed_dep = target_hyp[1]

    ind_unobserved = (
        variable_metadata[proposed_ind]["extra"]["sparsity_rate"]
        > constants.SPARSITY_FOR_UNOBS
    )
    dep_unobserved = (
        variable_metadata[proposed_dep]["extra"]["sparsity_rate"]
        > constants.SPARSITY_FOR_UNOBS
    )

    # if either are unobserved, we have no evidence that they are not correlated
    if ind_unobserved or dep_unobserved:
        return True, proposed_ind, proposed_dep
    # evidence of lack of correlation
    elif proposed_dep not in variable_metadata[proposed_ind]["corrs"]:
        return False, None, None
    # evidence of correlation
    else:
        return True, proposed_ind, proposed_dep


def determine_ctrl_vars(
    variable_metadata: Dict[str, Any],
    ind_var: str,
    dep_var: str,
    hypotheses: nx.DiGraph,
) -> Tuple[List[str], List[str]]:
    """Implements decision tree in Figure 1 from eval spec"""
    ctrl_vars = []
    not_ctrls = []
    for var in variable_metadata:
        if var in {ind_var, dep_var}:
            not_ctrls.append(var)
        elif are_correlated(var, dep_var, variable_metadata) or are_correlated(
            var, ind_var, variable_metadata
        ):
            ctrl_vars.append(var)
        elif are_correlated(var, dep_var, variable_metadata) is not None:
            # don't control vars which we have observed to be uncorrelated w/ dep
            not_ctrls.append(var)
        else:  # when dep_var or var is unobserved, no evidence of lack of correlation
            # control for any var which might influence the dependent variable
            dep_var_ancestors = nx.ancestors(hypotheses, dep_var)
            if var in dep_var_ancestors:
                ctrl_vars.append(var)
            else:
                not_ctrls.append(var)

    return ctrl_vars, not_ctrls


def are_correlated(var_1, var_2, variable_metadata) -> Optional[bool]:
    """
    Returns whether two variables are correlated. If there is no evidence
    of correlation, returns None.
    """
    if (
        variable_metadata[var_1]["extra"]["sparsity_rate"]
        > constants.SPARSITY_FOR_UNOBS
        or variable_metadata[var_2]["extra"]["sparsity_rate"]
        > constants.SPARSITY_FOR_UNOBS
    ):
        return None
    return (
        var_2 in variable_metadata[var_1]["corrs"]
        or var_1 in variable_metadata[var_2]["corrs"]
    )


def integrate_target_hyp(
    target_hyp: Tuple[Any, Any], hyp_graph: nx.DiGraph, np_rng: np.random.Generator
):
    """
    Integrates the target hypothesis into the hypotheses graph, respecting
    the original edge count by removing a random edge if necessary.
    """
    if not hyp_graph.has_edge(*target_hyp):
        random_edge_to_remove = np_rng.choice(list(hyp_graph.edges))
        hyp_graph.remove_edge(*random_edge_to_remove)
        hyp_graph.add_edge(*target_hyp)
    return hyp_graph


def gen_samples(
    n_samples: int,
    signal_noise_ratio: Optional[float],
    np_rng: np.random.Generator,
    balanced_ctrl_vars: bool = False,
) -> List[Sample]:
    samples = []
    if not balanced_ctrl_vars:
        for _ in tqdm(range(n_samples)):
            sample = gen_sample(signal_noise_ratio, np_rng)
            samples.append(sample)
    else:
        for _ in tqdm(range(n_samples)):
            sample = gen_sample_balanced_ctrl_vars(signal_noise_ratio, np_rng)
            samples.append(sample)

    return samples


def main(args: argparse.Namespace):
    np_rng = np.random.default_rng(args.seed)
    samples = gen_samples(args.n_samples, args.snr, np_rng, args.balanced_ctrl_vars)
    os.makedirs(args.jsonl_dir, exist_ok=True)
    if not args.balanced_ctrl_vars:
        jsonl_path = os.path.join(args.jsonl_dir, f"{args.n_samples}.jsonl")
    else:
        jsonl_path = os.path.join(
            args.jsonl_dir, f"{args.n_samples}_balanced_ctrl_vars.jsonl"
        )
    write_to_jsonl(samples, jsonl_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument(
        "--snr",
        type=float,
        default=None,
        help="signal-to-noise ratio. Default None (no noise is applied.)",
    )
    parser.add_argument(
        "--jsonl_dir", type=str, default="./evals/registry/data/identifying_variables/"
    )
    parser.add_argument("--seed", type=int, default=20220722)
    parser.add_argument(
        "--balanced_ctrl_vars",
        action="store_true",
        help="Whether to generate samples with balanced control variables.",
        default=False,
    )
    args = parser.parse_args()

    main(args)
