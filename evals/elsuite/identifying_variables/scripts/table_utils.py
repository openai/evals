from typing import Dict, List
from pathlib import Path

import numpy as np
import pandas as pd


def make_main_metric_table(
    results_dict: Dict,
    metric: str,
    solvers: List[str],
    renderers: List[str],
    save_dir: Path,
):
    """
    Makes and saves a table containing the information of performance of
    each solver for each renderer for each variant of the eval on
    a given metric.
      - Table rows are solvers; they are multi-rows, so each row has two subrows: with
        tree and without tree
      - Table columns are renderers; they are multi-columns, so each column has two
        subcolumns: mean and sem (standard error of the mean)

    Args:
        results_dict: dictionary containing the results of the eval. See
        `initialize_default_results_dict` and `populate_default_results_dict` in
        `process_results.py`.
        metric: the name of the metric we want to make the table for
        solvers: list of solvers we want to include in the table
        renderers: list of renderers we want to include in the table
        save_dir: directory to save the table in (as a CSV file)
    """

    # only keep keep metric in results_dict
    filtered_results_dict = results_dict[metric]
    # flatten into tuples
    data_tuples = []
    for stat, solver_data in filtered_results_dict.items():
        for solver, renderer_data in solver_data.items():
            for renderer, tree_data in renderer_data.items():
                for tree_type, value in tree_data.items():
                    if value is not None:
                        data_tuples.append((solver, tree_type, renderer, stat, value))

    df = pd.DataFrame(
        data_tuples, columns=["Solver", "Tree", "Renderer", "Stat", "Value"]
    )
    df = df.pivot_table(
        index=["Solver", "Tree"], columns=["Renderer", "Stat"], values="Value"
    )
    # sorting by solvers, renderers (for some reason ordering is lost in the above process)
    new_index = [
        (solver, tree) for solver in solvers for tree in ["with tree", "without tree"]
    ]
    new_columns = pd.MultiIndex.from_product(
        [renderers, df.columns.levels[1]], names=df.columns.names
    )
    df = df.reindex(new_index, columns=new_columns)

    # delete the with tree rows for the treeless solvers
    for solver in solvers[-2:]:
        df.drop((solver, "with tree"), inplace=True)

    # save table
    save_path = save_dir / f"{metric}_table.csv"
    df.to_csv(save_path)
