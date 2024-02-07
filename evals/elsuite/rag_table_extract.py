import os
import traceback
from io import StringIO
import json
import re
from pathlib import Path

from typing import List, Optional, Tuple, Union

import pandas as pd
import numpy as np
from pydantic import BaseModel
import uuid

import evals
import evals.metrics
from evals.api import CompletionFn
from evals.elsuite.rag_match import get_rag_dataset
from evals.elsuite.utils import fuzzy_compare, fuzzy_normalize_name, tableMatching, tableMatchingStrict
from evals.record import RecorderBase, record_match


code_pattern = r"```[\s\S]*?\n([\s\S]+?)\n```"
json_pattern = r"```json[\s\S]*?\n([\s\S]+?)\n```"
csv_pattern = r"```csv[\s\S]*?\n([\s\S]+?)\n```"
outlink_pattern = r"\[Download[a-zA-Z0-9 ]+?\]\((https://[a-zA-Z0-9_. /]+?)\)"


def parse_csv_text(csvtext: str) -> str:
    lines = csvtext.strip().split("\n")
    tuple_pattern = r"\((\"[\s\S]*?\"),(\"[\s\S]*?\")\)"
    if re.search(tuple_pattern, lines[0]) is not None:
        lines[0] = re.sub(tuple_pattern, r"(\1|\2)", lines[0])
    lines_clr = [re.sub(r"\"[\s\S]*?\"", "", line) for line in lines]
    max_commas = max([line_clr.count(",") for line_clr in lines_clr])
    unified_lines = [line + ("," * (max_commas - line_clr.count(","))) for line, line_clr in zip(lines, lines_clr)]
    return "\n".join(unified_lines)


def parse_table_multiindex(table: pd.DataFrame, compare_fields: list = []) -> pd.DataFrame:
    """
    Parse a table with multiindex columns.
    """

    df = table.copy()
    if df.columns.nlevels == 1 and tuple in [type(f) for f in compare_fields]:
        coltypes = {col: type(df[col].iloc[0]) for col in df.columns}
        for col, ctype in coltypes.items():
            if ctype == str:
                if ":" in df[col].iloc[0] and "," in df[col].iloc[0]:
                    df[col] = [{key: value for key, value in [pair.split(": ") for pair in data.split(", ")]} for data
                               in df[col]]
                    coltypes[col] = dict
        dfs = []

        for col, ctype in coltypes.items():
            if ctype == dict:
                d = pd.DataFrame(df.pop(col).tolist())
                d.columns = pd.MultiIndex.from_tuples([(col, fuzzy_normalize_name(key)) for key in d.columns])
                dfs.append(d)
        df.columns = pd.MultiIndex.from_tuples([eval(col.replace("|", ",")) if (col[0] == "(" and col[-1] == ")") else
                                                (col, "") for col in df.columns])
        df = pd.concat([df] + dfs, axis=1)
    if df.columns.nlevels > 1:
        df.columns = pd.MultiIndex.from_tuples([(col, fuzzy_normalize_name(subcol)) for col, subcol in df.columns])

    return df


class FileSample(BaseModel):
    file_name: Optional[str]
    file_link: Optional[str]
    answerfile_name: Optional[str]
    answerfile_link: Optional[str]
    compare_fields: List[Union[str, Tuple]]
    index: Union[str, Tuple] = ("Compound", "")


class TableExtract(evals.Eval):
    def __init__(
            self,
            completion_fns: list[CompletionFn],
            samples_jsonl: str,
            *args,
            instructions: Optional[str] = "",
            **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert len(completion_fns) < 3, "TableExtract only supports 3 completion fns"
        self.samples_jsonl = samples_jsonl
        self.instructions = instructions

    def eval_sample(self, sample, rng):
        assert isinstance(sample, FileSample)

        prompt = \
                self.instructions
                # + f"\nThe fields should at least contain {sample.compare_fields}"
        result = self.completion_fn(
            prompt=prompt,
            temperature=0.0,
            max_tokens=5,
            file_name=sample.file_name,
            file_link=sample.file_link
        )
        sampled = result.get_completions()[0]

        compare_fields_types = [type(x) for x in sample.compare_fields]
        header_rows = [0, 1] if tuple in compare_fields_types else [0]

        correct_answer = parse_table_multiindex(pd.read_csv(sample.answerfile_name, header=header_rows).astype(str), compare_fields=sample.compare_fields)
        correct_answer.to_csv("temp.csv", index=False)
        correct_str = open("temp.csv", 'r').read()

        try:
            if re.search(outlink_pattern, sampled) is not None:
                code = re.search(outlink_pattern, sampled).group()
                link = re.sub(outlink_pattern, r"\1", code)

                fname = f"/tmp/LLMEvals_{uuid.uuid4()}.csv"
                os.system(f"wget {link} -O {fname}")
                table = pd.read_csv(fname)
                if pd.isna(table.iloc[0, 0]):
                    table = pd.read_csv(fname, header=header_rows)
            elif "csv" in prompt:
                code = re.search(csv_pattern, sampled).group()
                code_content = re.sub(csv_pattern, r"\1", code)
                code_content_processed = parse_csv_text(code_content)
                # table = pd.read_csv(StringIO(code_content_processed), header=header_rows)
                table = pd.read_csv(StringIO(code_content_processed))
                if pd.isna(table.iloc[0, 0]):
                    table = pd.read_csv(StringIO(code_content_processed), header=header_rows)

            elif "json" in prompt:
                code = re.search(json_pattern, sampled).group()
                code_content = re.sub(json_pattern, r"\1", code).replace("\"", "")
                table = pd.DataFrame(json.loads(code_content))
            else:
                table = pd.DataFrame()
            table = parse_table_multiindex(table, compare_fields=sample.compare_fields)

            if sample.index not in table.columns:
                table.columns = [sample.index] + list(table.columns)[1:]
            answerfile_out = sample.answerfile_name.replace(".csv", "_output.csv")
            table.to_csv(answerfile_out, index=False)
            picked_str = open(answerfile_out, 'r').read()
        except:
            print(Path(sample.file_name).stem)
            traceback.print_exc()
            record_match(
                prompt=prompt,
                correct=False,
                expected=correct_str,
                picked=sampled,
                file_name=sample.file_name,
                jobtype="match_all"
            )
            return

        metrics = tableMatching(correct_answer, table, index=sample.index, compare_fields=sample.compare_fields,
                                record=False, file_name=sample.file_name)
        record_match(
            prompt=prompt,
            correct=(metrics["recall_field"] == 1.0 and metrics["recall_index"] == 1.0 and metrics["recall_value"] == 1.0),
            expected=correct_str,
            picked=picked_str,
            file_name=sample.file_name,
            jobtype="match_all"
        )

    def run(self, recorder: RecorderBase):
        raw_samples = get_rag_dataset(self._prefix_registry_path(self.samples_jsonl).as_posix())
        for raw_sample in raw_samples:
            raw_sample["compare_fields"] = [field if type(field) == str else tuple(field) for field in
                                            raw_sample["compare_fields"]]

        samples = [FileSample(**raw_sample) for raw_sample in raw_samples]
        metrics_all_sample = self.eval_all_samples(recorder, samples)

        metrics = {key: np.mean([sample_metrics[key] for sample_metrics in metrics_all_sample]) for key in
                   ["recall_field", "recall_index", "recall_value", "recall_value_strict", "accuracy_value", "accuracy_value_strict"]}
        if "SMILES" in raw_samples[0]["compare_fields"]:
            metrics["recall_SMILES"] = np.mean([sample_metrics["recall_SMILES"] for sample_metrics in metrics_all_sample
                                                if "recall_SMILES" in sample_metrics])
        return metrics
        # return {
        #     "accuracy": evals.metrics.get_accuracy(recorder.get_events("match")),
        # }
