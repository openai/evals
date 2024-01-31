import argparse
import json
import pickle
import re
import glob
from io import StringIO
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def parse_jsonl_log(logfile):
    with open(logfile, "r") as f:
        events_df = pd.read_json(f, lines=True)
    if not "final_report" in events_df.columns:
        return
    logger_data = final_report = events_df["final_report"].dropna().iloc[0]

    print(events_df)
    run_config = events_df.loc[0, "spec"]
    evalname = run_config["base_eval"]
    model = run_config["completion_fns"][0].replace("/", ".")
    matches_df = events_df[events_df["type"] == "match"].reset_index(drop=True)
    matches_df = matches_df.join(pd.json_normalize(matches_df.data))
    matches_df.pop("data")
    if "options" in matches_df.columns:
        matches_df["options"] = matches_df["options"].astype(str)
    matches_df["created_at"] = matches_df["created_at"].astype(str)

    matches_df.insert(0, "model", model)
    matches_df.insert(0, "evalname", evalname)

    if "file_name" in matches_df.columns:
        matches_df.insert(2, "doi", [re.sub("__([0-9]+)__", r"(\1)", Path(f).stem).replace("_", "/")
                                     for f in matches_df["file_name"]])

        # TODO: compare on different completion_functions
        if "jobtype" in matches_df.columns:
            # Table extract tasks
            accuracy_by_type_and_file = matches_df.groupby(["jobtype", "doi"])['correct'].mean().reset_index()
            accuracy_by_type_and_file["model"] = model

            accuracy_by_type = matches_df.groupby(["jobtype"])['correct'].mean().to_dict()
            print(accuracy_by_type_and_file)

            logger_data = {**logger_data,
                           **{f"Accuracy_{key}": value for key, value in accuracy_by_type.items()}}

            for doi, df in matches_df.groupby("doi"):
                print(df)
                logger_data[f"{doi.replace('/', '_')}/context:match"] = df[df["jobtype"] != "match_all"][
                    ["correct", "expected", "picked", "jobtype"]]
                match_all_data = df[df["jobtype"] == "match_all"].iloc[0, :]
                logger_data[f"{doi.replace('/', '_')}/context:truth"] = pd.read_csv(
                    StringIO(match_all_data["expected"]), header=[0, 1])
                logger_data[f"{doi.replace('/', '_')}/context:extract"] = pd.read_csv(
                    StringIO(match_all_data["picked"]), header=[0, 1]) \
                    if df["jobtype"].iloc[0] != "match_all" else match_all_data["picked"]
        else:
            accuracy_by_type_and_file = None
    else:
        matches_df.insert(2, "doi", "")
        accuracy_by_type_and_file = None
    # Regular tasks
    matches_df_formatted = matches_df.dropna(subset="prompt")
    if type(matches_df_formatted["prompt"].iloc[0]) == list:
        matches_df_formatted["prompt"] = ["\n\n".join([f"**{message['role']}**: {message['content']}" for message in prompt])
                                          for prompt in matches_df_formatted["prompt"]]
    logger_data["evals_table"] = matches_df_formatted

    return run_config, evalname, model, final_report, logger_data, accuracy_by_type_and_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Report evals results")
    parser.add_argument("run_id", type=str, nargs="+", help="Eval Run id")
    parser.add_argument("--mlops", type=str, default=None)
    parser.add_argument("--name", type=str, default="LLM_Eval")

    args = parser.parse_args()

    logfiles = []
    for run_id in args.run_id:
        logfiles += glob.glob(f"/tmp/evallogs/{run_id}*/**", recursive=True) + glob.glob(f"/tmp/evallogs/{run_id}*.jsonl")
    logfiles = sorted([f for f in logfiles if Path(f).suffix == ".jsonl"])
    logger_data = {}
    table_collection = []
    qa_collection = []

    for logfile in logfiles:
        run_config, evalname, model, final_report, logger_data, accuracy_by_type_and_file = parse_jsonl_log(logfile)
        qa_collection.append({"evalname": evalname, "model": model, **final_report})
        if accuracy_by_type_and_file:
            table_collection.append(accuracy_by_type_and_file)

    if len(table_collection) > 0:
        print("Logging TableExtraction Metrics...")
        accuracy_by_model_type_and_file = pd.concat(table_collection)
        metrics_by_eval = pd.DataFrame(qa_collection)
        accuracies = metrics_by_eval[metrics_by_eval["accuracy"] >= 0]

        if "score" in metrics_by_eval.columns:
            scores = metrics_by_eval[metrics_by_eval["score"] >= 0]

        if args.mlops:
            import plotly.express as px
            logger_data["TableExtraction"] = px.box(accuracy_by_model_type_and_file,
                                                    x="jobtype", y="correct", color="model",
                                                    title="Accuracy by jobtype and model")
            logger_data["QA_accuracy"] = px.bar(accuracies, x="eval", y="accuracy", color="model",
                                                title="Accuracy by eval and model")
            if "score" in metrics_by_eval.columns:
                logger_data["QA_score"] = px.bar(scores, x="eval", y="accuracy", color="model",
                                                 title="Accuracy by eval and model")
    if args.mlops:
        config_logger = json.load(open(args.mlops, 'r'))
        if "name" not in config_logger.keys():
            config_logger["name"] = args.name
        if "dp_mlops" in config_logger:
            from evals.reporters.DPTracking import DPTrackingReporter
            DPTrackingReporter.report_run(config_logger, {}, logger_data, step=0)


if __name__ == "__main__":
    main()
