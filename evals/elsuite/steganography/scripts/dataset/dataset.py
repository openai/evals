import csv
from pathlib import Path

import pandas as pd
from custom_datasets import (
    HFTaskDataset,
    HFTextDataset,
    RandomCharAndNumberDataset,
    RandomWordsDataset,
)


def make_task_data():
    # Task data
    n_samples_per_dataset = 160

    def mmlu_prompt_func(row):
        prompt = f"{row['question']}\n"
        for idx, choice in enumerate(row["choices"]):
            prompt += f"{idx}: {choice}\n"
        return prompt

    datasets = [
        HFTaskDataset(
            hf_path="alespalla/chatbot_instruction_prompts",
            dataset_name="",
            split="train",
            n_samples=n_samples_per_dataset,
            prompt_func=lambda row: f"{row['prompt']}",
            output_func=lambda row: row["response"],
        ),
        HFTaskDataset(
            hf_path="akoksal/LongForm",
            dataset_name="",
            split="train",
            n_samples=n_samples_per_dataset,
            prompt_func=lambda row: f"{row['input']}",
            output_func=lambda row: row["output"],
        ),
        HFTaskDataset(
            hf_path="lighteval/mmlu",
            dataset_name="all",
            split="dev",
            n_samples=n_samples_per_dataset,
            prompt_func=mmlu_prompt_func,
            output_func=lambda row: row["answer"],
        ),
    ]

    df_rows = []
    for dset in datasets:
        for sample in dset:
            df_rows.append(sample)

    df = pd.DataFrame(df_rows)
    # Summary stats
    print(df.groupby("type").agg({"length": ["mean", "std", "min", "max"]}))
    return df


def make_payload_data():

    # Payload data
    n_samples_per_dataset = 96
    max_tokens_per_doc = 512
    random_dataset_lengths = [3, 5, 10, 20]
    datasets = [
        HFTextDataset(
            hf_path="Abirate/english_quotes",
            dataset_name="",
            split="train",
            n_samples=n_samples_per_dataset,
            max_tokens_per_doc=max_tokens_per_doc,
            text_field="quote",
        ),
        HFTextDataset(
            hf_path="c4",
            dataset_name="en",
            split="validation",
            n_samples=n_samples_per_dataset,
            max_tokens_per_doc=max_tokens_per_doc,
            text_field="text",
        ),
        HFTextDataset(
            hf_path="wikipedia",
            dataset_name="20220301.en",
            split="train",
            n_samples=n_samples_per_dataset,
            max_tokens_per_doc=max_tokens_per_doc,
            text_field="title",
        ),
        RandomCharAndNumberDataset(n_samples=n_samples_per_dataset, lengths=random_dataset_lengths),
        RandomWordsDataset(n_samples=n_samples_per_dataset, lengths=random_dataset_lengths),
    ]

    df_rows = []
    for dset in datasets:
        for sample in dset:
            df_rows.append(sample)

    df = pd.DataFrame(df_rows)
    # Summary stats
    print(df.groupby("type").agg({"length": ["mean", "std", "min", "max"]}))
    return df


if __name__ == "__main__":
    random_seed = 0
    task_df = make_task_data()
    print(task_df)
    payload_df = make_payload_data()
    print(payload_df)

    assert len(task_df) == len(
        payload_df
    ), "Task and payload datasets must have the same number of samples"

    # Merge datasets
    # add prefix to column names
    task_df = task_df.add_prefix("task_")
    payload_df = payload_df.add_prefix("payload_")
    # shuffle each first
    task_df = task_df.sample(frac=1, random_state=random_seed)
    payload_df = payload_df.sample(frac=1, random_state=random_seed)
    # Join each (task, payload) as a row
    df = pd.concat([task_df, payload_df], axis=1)

    print(df)

    outpath = Path("dataset.csv")
    print(f"Saving {len(df)} samples to {outpath}")
    df.to_csv(outpath, index=False, quoting=csv.QUOTE_ALL)
