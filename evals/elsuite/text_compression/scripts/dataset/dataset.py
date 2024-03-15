import csv

import pandas as pd
from custom_datasets import HFTextDataset, RandomCharAndNumberDataset, RandomWordsDataset

if __name__ == "__main__":
    n_samples_per_dataset = 50
    max_tokens_per_doc = 2048
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
        ),
        HFTextDataset(
            hf_path="openwebtext",
            dataset_name="plain_text",
            split="train",
            n_samples=n_samples_per_dataset,
            max_tokens_per_doc=max_tokens_per_doc,
        ),
        HFTextDataset(
            hf_path="oscar",
            dataset_name="unshuffled_deduplicated_en",
            split="train",
            n_samples=n_samples_per_dataset,
            max_tokens_per_doc=max_tokens_per_doc,
        ),
        HFTextDataset(
            hf_path="wikipedia",
            dataset_name="20220301.en",
            split="train",
            n_samples=n_samples_per_dataset,
            max_tokens_per_doc=max_tokens_per_doc,
        ),
        HFTextDataset(
            hf_path="codeparrot/github-code",
            dataset_name=None,
            split="train",
            licenses=["mit"],
            n_samples=n_samples_per_dataset,
            max_tokens_per_doc=max_tokens_per_doc,
            text_field="code",
        ),
        RandomCharAndNumberDataset(n_samples=n_samples_per_dataset),
        RandomWordsDataset(n_samples=n_samples_per_dataset),
    ]

    df_rows = []
    for dset in datasets:
        for sample in dset:
            df_rows.append(sample)

    df = pd.DataFrame(df_rows)
    print(f"Saving dataset.csv with {len(df)} samples")
    df.to_csv("dataset.csv", index=False, quoting=csv.QUOTE_ALL)

    # Summary stats
    print(df.groupby("type").agg({"length": ["mean", "std", "min", "max"]}))
