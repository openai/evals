import re
from typing import Dict

import networkx as nx
import numpy as np

from evals.elsuite.identifying_variables.structs import Answer, Sample
from evals.solvers.solver import SolverResult


def parse_solver_preds(solver_result: SolverResult) -> Answer:
    solver_string = solver_result.output.strip().lower()

    pattern = (
        r"\[@answer "  # Matches the beginning of the answer
        r"valid_hyp: (true|false|True|False)"  # valid hyp part
        r"(?:; independent: ([^;]*))?"  # Optionally matches the independent part
        r"(?:; dependent: ([^;]*))?"  # Optionally matches the dependent part
        r"(?:; control: ([^\]]*))?"  # Optionally matches the control part
        r"\]"  # Matches the end of the answer
    )

    match = re.search(pattern, solver_string)

    if match:
        valid_hyp = match.group(1).lower() == "true"
        if not valid_hyp:
            return Answer(
                valid_hypothesis=False,
                ind_var=None,
                dep_var=None,
                ctrl_vars=None,
            )
        ind_var = match.group(2)
        ind_var = ind_var if ind_var is not None else "WRONG"
        dep_var = match.group(3)
        dep_var = dep_var if dep_var is not None else "WRONG"
        ctrl_vars = match.group(4)
        if ctrl_vars is not None:
            ctrl_vars = ctrl_vars.split(",")
            ctrl_vars = [var.strip() for var in ctrl_vars]
            if ctrl_vars[0].lower().strip("\"'`«»<>") == "none":
                ctrl_vars = []
        else:
            ctrl_vars = ["WRONG"]
        return Answer(
            valid_hypothesis=True,
            ind_var=ind_var,
            dep_var=dep_var,
            ctrl_vars=ctrl_vars,
        )
    else:
        raise ValueError("Invalid solver output")


def sample_serializer(obj):
    """
    Custom serializer to pass to json.dumps when
    saving a sample dictionary to jsonl
    """
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, nx.DiGraph):
        return nx.to_dict_of_lists(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)


def json_to_sample(serialized_sample: Dict) -> Sample:
    """Reads sample from jsonl into Sample dataclass"""
    hypotheses = nx.from_dict_of_lists(serialized_sample["hypotheses"], create_using=nx.DiGraph)
    causal_graph = nx.from_dict_of_lists(serialized_sample["causal_graph"], create_using=nx.DiGraph)
    gold_label = Answer(**serialized_sample["gold_label"])

    # convert corrs in variable_metadata from lists to sets
    for var in serialized_sample["variable_metadata"]:
        serialized_sample["variable_metadata"][var]["corrs"] = set(
            serialized_sample["variable_metadata"][var]["corrs"]
        )

    return Sample(
        variable_metadata=serialized_sample["variable_metadata"],
        hypotheses=hypotheses,
        target_hypothesis=serialized_sample["target_hypothesis"],
        sample_metadata=serialized_sample["sample_metadata"],
        causal_graph=causal_graph,
        gold_label=gold_label,
        num_not_ctrl=serialized_sample["num_not_ctrl"],
    )
