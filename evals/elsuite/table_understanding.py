import csv
from io import StringIO

import evals
import evals.metrics
from evals.api import CompletionFn
from evals.record import RecorderBase


class TableUnderstanding(evals.Eval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        samples_jsonl: str,
        *args,
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert len(completion_fns) == 1, "Includes only supports one completion fn"
        self.samples_jsonl = samples_jsonl

    def run(self, recorder: RecorderBase):
        """
        Called by the `oaieval` CLI to run the eval.
        The `eval_all_samples` method calls `eval_sample`.
        """
        samples = self.get_samples()
        self.eval_all_samples(recorder, samples)

        accuracies = recorder.get_scores("accuracy")
        return {"accuracy": sum(accuracies) / len(accuracies)}

    def eval_sample(self, sample, *_):
        prompt, correct_answers = sample["input"], sample["ideal"]

        result = self.completion_fn(
            prompt=prompt,
        )
        generated_answer = result.get_completions()[0]
        accuracy = eval_table(generated_answer, correct_answers)
        print(accuracy)

        evals.record.record_metrics(
            accuracy=accuracy,
        )


def get_tbl_rows(tbl: str):
    csv_reader = csv.reader(StringIO(tbl), delimiter=",")

    rows = []
    for row in csv_reader:
        rows.append(row)

    return rows


def matching_rows(rows_pred, rows_true):
    n_matches = 0

    for row_true, row_pred in zip(rows_true, rows_pred):
        # first check if prices match
        if row_true[-1] != row_pred[-1]:
            continue
        # count number of matching values in the row
        for val_true, val_pred in zip(row_true, row_pred):
            if val_true == val_pred:
                n_matches += 1
    return n_matches


def eval_table(tbl_pred: str, tbl_true: str):
    rows_pred = get_tbl_rows(tbl_pred)
    rows_true = get_tbl_rows(tbl_true)

    if len(rows_pred) < 2:
        return 0
    # discard all rows which are most probably not related to output csv
    relevant_rows = [row for row in rows_pred if len(row) == 7]
    rows_pred = sorted(relevant_rows, key=lambda x: x[-1])[1:]
    rows_true = sorted(rows_true, key=lambda x: x[-1])[1:]

    n_matches = matching_rows(rows_pred, rows_true)
    n_values = sum([len(row) for row in rows_true])
    return n_matches / n_values
