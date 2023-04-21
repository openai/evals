import csv
import io
import random
import re

import evals
import evals.metrics
from evals.record import Recorder
from sklearn.model_selection import train_test_split

class CSVFormatWithRowsColumnsAndData(evals.Eval):
    def __init__(self, model_spec, data_jsonl, train_samples_per_prompt=2, test_size=0.8, random_state=None, **kwargs):
        super().__init__(**kwargs)
        self.model_spec = model_spec
        self.data_jsonl = data_jsonl
        self.train_samples_per_prompt = train_samples_per_prompt
        self.test_size = test_size
        self.random_state = random_state

        self.train_samples, self.test_samples = self.load_and_split_data()
        
    def load_and_split_data(self):
        data_samples = evals.get_jsonl(self.data_jsonl)
        train_samples, test_samples = train_test_split(data_samples, test_size=self.test_size, random_state=self.random_state)
        return train_samples, test_samples

    def run(self, recorder):
        self.eval_all_samples(recorder, self.test_samples)
        
        evaluation_results = self.evaluate(recorder)

        return evaluation_results

    def eval_sample(self, test_sample, rng: random.Random, recorder: Recorder):
        stuffing = rng.sample(self.train_samples, self.train_samples_per_prompt)

        prompt = [
            {"role": "system", "content": '''You are a language model that only outputs CSV strings without a header and within triple backticks.'''},
        ]

        for i, sample in enumerate(stuffing + [test_sample]):
            if i < len(stuffing):
                prompt += [
                    {"role": "system", "content": sample["prompt"], "name": "example_user"},
                    {"role": "system", "content": sample["expected_output"], "name": "example_assistant"},
                ]
            else:
                prompt += [{"role": "user", "content": sample["prompt"]}]

        # Get the output from the model
        output = self.model_spec(prompt)
        
        # Extract the content from the model's output
        content = output.raw_data.choices[0].message.content.strip() if hasattr(output, "raw_data") else ""

        # Extract the CSV content from the content
        csv_content = re.search(r'```\s*(.+?)\s*```', content.replace('\r', ''), re.DOTALL)
        if csv_content:
            csv_output = csv_content.group(1)
        else:
            csv_output = content
        
        #print(f'csv_content: {csv_content}')
        #print(f'csv_output: {csv_output}')
        
        # Extract the number of rows and columns from the prompt
        rows, columns = map(int, re.findall(r'\d+', test_sample["prompt"]))
        #print(f'expected rows: {rows}\nexpected columns: {columns}')

        # Use check_csv_output with the output from the model
        results = self.check_csv_output(csv_output, expected_rows=rows, expected_columns=columns)

        recorder.record_event("match", {
            "format_match": results["format_match"],
            "row_count_match": results["row_count_match"],
            "col_count_match": results["col_count_match"],
            "cell_data_match": results["cell_data_match"]
        })

        return {
            "output": csv_output,
            "meta": {},
            "format_match": results["format_match"],
            "row_count_match": results["row_count_match"],
            "col_count_match": results["col_count_match"],
            "cell_data_match": results["cell_data_match"]
        }

    def check_csv_output(self, output, expected_rows, expected_columns):
        results = {
            "format_match": True,
            "row_count_match": True,
            "col_count_match": True,
            "cell_data_match": True
        }

        try:
            reader = csv.reader(io.StringIO(output), skipinitialspace=True)
            rows = [row for row in reader]
        except (csv.Error, AttributeError):
            results["format_match"] = False
            results["row_count_match"] = False
            results["col_count_match"] = False
            results["cell_data_match"] = False
            return results

        if len(rows) != int(expected_rows):
            results["row_count_match"] = False

        for row in rows:
            if len(row) != expected_columns:
                results["col_count_match"] = False
            elif any(cell.strip() == "" for cell in row):
                results["cell_data_match"] = False

        return results

        
    def evaluate(self, recorder):
        # Calculate the number of matches for each metric
        format_matches = 0
        row_count_matches = 0
        col_count_matches = 0
        cell_data_matches = 0

        events = recorder.get_events("match")

        for event in events:
            if event.data["format_match"]:
                format_matches += 1
            if event.data["row_count_match"]:
                row_count_matches += 1
            if event.data["col_count_match"]:
                col_count_matches += 1
            if event.data["cell_data_match"]:
                cell_data_matches += 1

        total_samples = len(events)

        # Handle the case when total_samples is zero
        if total_samples == 0:
            return {
                "format_accuracy": 0,
                "row_count_accuracy": 0,
                "col_count_accuracy": 0,
                "cell_data_accuracy": 0,
            }

        # Calculate the percentage of matches for each metric
        format_accuracy = format_matches / total_samples
        row_count_accuracy = row_count_matches / total_samples
        col_count_accuracy = col_count_matches / total_samples
        cell_data_accuracy = cell_data_matches / total_samples

        return {
            "format_accuracy": format_accuracy,
            "row_count_accuracy": row_count_accuracy,
            "col_count_accuracy": col_count_accuracy,
            "cell_data_accuracy": cell_data_accuracy,
        }

def eval_all_samples(self, recorder, samples):
    rng = random.Random()
    rng.seed(self.random_seed)

    results = []
    for idx, sample in enumerate(samples):
        result = self.eval_sample(sample, rng, recorder)
        results.append((idx, result))

    return results
