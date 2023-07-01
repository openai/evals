# -*- coding: utf-8 -*-
""" Just some view modifier for a quick visual analysis
"""

import jsonlines

import pandas as pd
from box import Box

def load_result_to_DataFrame(out_path: str):
    global line, value, df
    ## Analytical re-processing, it only works when you have the out_path file generated from previous run,
    # ie. the main() is disabled:
    lines = {}  # to be indexed by run_id
    with jsonlines.open(out_path) as reader:
        # parsing run_id-connected stuff to a DF
        for line in reader:
            if "run_id" in line:
                # interested, grab from it:
                if line["type"] == "sampling":
                    lines[line["sample_id"]] = {"statement": Box(line).data.prompt[1].content}
                if line["type"] == "match":
                    lines[line["sample_id"]].update(Box(line).data)
    data_list = [{'run_id': key, **value} for key, value in lines.items()]
    return pd.DataFrame(data_list)
