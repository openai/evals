"""Custom data structures for the eval"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx


@dataclass
class Answer:
    valid_hypothesis: bool
    ind_var: Optional[str]
    dep_var: Optional[str]
    ctrl_vars: Optional[List[str]]


@dataclass
class Sample:
    """
    A sample of the dataset for the eval.

    Args:
        variable_metadata (Dict) : A dictionary mapping each variable name to its metadata.
            Each variable's metadata is a dictionary containing:
                - 'gen_method': A dictionary specifying the generation method for the
                   variable, including:
                    - 'name': Name of the latent function or distribution.
                    - 'input_x': Name of the input variable, if applicable.
                    - 'kwargs': Additional arguments for the latent function.
                - 'corrs': A set of variables correlated with this variable.
        hypotheses (nx.DiGraph): A directed acyclic graph (DAG) representing the hypotheses.
        target_hypothesis (Tuple[str, str]) A tuple (independent_variable, dependent_variable)
            representing the hypothesis of interest.
        sample_metadata (Dict): A dictionary with additional metadata, including:
            - 'num_obs_samples': Number of observations generated per variable.
            - 'snr': Signal-to-noise ratio applied to the observations.
        causal_graph (nx.DiGraph): A randomly generated DAG representing the underlying
            causal relationships among variables. Represented as nx.DiGraph.
        gold_label (Answer): The gold label for the sample.
        num_not_ctrl (Optional[int]): The number of variables not controlled for. None
            if the hypothesis is invalid.
    """

    variable_metadata: Dict
    hypotheses: nx.DiGraph
    target_hypothesis: Tuple[str, str]
    sample_metadata: Dict
    causal_graph: nx.DiGraph
    gold_label: Answer
    num_not_ctrl: Optional[int]
