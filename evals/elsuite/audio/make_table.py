import argparse
from pathlib import Path
from urllib import parse

import pandas as pd

from evals.utils import log_utils


def extract_results(datadir: Path) -> pd.DataFrame:
    df_rows = []
    for path, results in log_utils.get_final_results_from_dir(datadir).items():
        spec = log_utils.extract_spec(path)
        base_eval = spec["base_eval"]
        model = spec["completion_fns"][0].split("/")[-1]
        max_samples = spec["run_config"]["max_samples"]
        ds_url = spec["run_config"]["eval_spec"]["args"]["dataset"]
        ds_args = parse.parse_qs(parse.urlparse(ds_url).query)
        ds_subset = ds_args.get("name", ["-"])[0]
        obj = {
            "eval": base_eval,
            "subset": ds_subset,
            "model": model,
            "samples": max_samples,
            "score": results.get("sacrebleu_score")
            or results.get("wer", 0) * 100
            or results.get("score", 0) * 100
            or results.get("accuracy", 0) * 100,
            "tokens": results.get("usage_total_tokens"),
        }
        df_rows.append(obj)
    df_rows.sort(key=lambda x: (x["eval"], x["model"]))
    df = pd.DataFrame(df_rows)
    pd.set_option("display.precision", 2)  # Adjust precision as needed
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", "-d", type=str, required=True)
    parser.add_argument("--out_dir", "-o", type=str, default="./outputs")
    args = parser.parse_args()
    log_dir = Path(args.log_dir)

    df = extract_results(log_dir)
    print(df)


if __name__ == "__main__":
    main()
