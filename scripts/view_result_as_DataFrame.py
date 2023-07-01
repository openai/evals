# -*- coding: utf-8 -*-
""" Just some view modifier for a quick visual analysis
"""

import jsonlines

import pandas as pd
from box import Box

def get_latest_file(path, pattern):
    """ Cool to use with both `print_log` and `load_result_to_DataFrame` """
    import os, glob
    list_of_files = glob.glob(os.path.join(path, pattern))
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def print_log(log_output):
    """ give it for example log_output="/tmp/evallogs/230530205047OSMJXQ5A_gpt-3.5-turbo_logic-fact.jsonl """
    with open(log_output,"r") as fp:
        import json
        for line in fp.readlines():
          json_object = json.loads(line)
          json_formatted_str = json.dumps(json_object, indent=2)
          print(json_formatted_str)


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

######### Zone of doubt (maybe not useful code)
######### ::::::::::::::::::::
def display_table_from_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Parse each line as a JSON object
            json_line = json.loads(line)

            # Extract 'content' from 'input' and 'ideal' fields
            content = json_line['input'][0]['content']
            ideal = json_line['ideal']

            # Append to data list
            data.append({'Content': content, 'Ideal': ideal})

    # Convert the list to a DataFrame
    df = pd.DataFrame(data)

    # Return the DataFrame
    return df


def display_outs():
    # !cd evals && OPENAI_API_KEY=$openai_key oaieval gpt-3.5-turbo logic-fact

    """Zobrazi evaly pro logic ve forme tabulky"""

    # Call the function with your file path and print the DataFrame
    df = display_table_from_file('evals/evals/registry/data/logic/samples.jsonl')
    display(df)

    """Zobrazi posledni vygenerovany log z **oaieval** pro logic-fact"""

    log_output=get_latest_file("/tmp/evallogs", "*gpt-3.5-turbo_logic-fact.jsonl")
    print("Log name:", log_output)
    print_log(log_output)

######### // End of one of doubt (maybe not useful code) ##############
########################################################################

