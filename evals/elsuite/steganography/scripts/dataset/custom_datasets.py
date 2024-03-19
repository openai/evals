import pickle
import string
from pathlib import Path

import complexity_metrics as complexity
import nltk
import numpy as np
import utils
from datasets import load_dataset

"""
Basic text datasets
"""


class BaseTextDataset:
    def __init__(self, seed=0):
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.dataset = []
        self.type = self.__class__.__name__

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Handling for a slice object
            start = idx.start if idx.start is not None else 0
            stop = idx.stop if idx.stop is not None else len(self.dataset)
            step = idx.step if idx.step is not None else 1
            return [
                {
                    "text": self.dataset[i],
                    "type": self.type,
                    "length": len(self.dataset[i]),
                    "entropy": complexity.calculate_entropy(self.dataset[i]),
                    "gzip_compression_ratio": complexity.calculate_compression_ratio(
                        self.dataset[i]
                    ),
                    "brevity_score": complexity.calculate_brevity_score(self.dataset[i]),
                }
                for i in range(start, stop, step)
            ]
        else:
            # Handling for a plain index
            return {
                "text": self.dataset[idx],
                "type": self.type,
                "length": len(self.dataset[idx]),
                "entropy": complexity.calculate_entropy(self.dataset[idx]),
                "gzip_compression_ratio": complexity.calculate_compression_ratio(self.dataset[idx]),
                "brevity_score": complexity.calculate_brevity_score(self.dataset[idx]),
            }


class HFTextDataset(BaseTextDataset):
    def __init__(
        self,
        hf_path,
        dataset_name,
        split,
        n_samples,
        streaming=True,
        seed=0,
        cache_dir="/tmp/hf_cache",
        max_tokens_per_doc=2048,
        text_field="text",
        use_cache=False,
        **kwargs,
    ):
        super().__init__(seed=seed)
        self.type = hf_path

        cache_id = f"{hf_path}_{dataset_name}_{split}_{n_samples}_{streaming}_{seed}"
        cache_path = Path(cache_dir) / f"{cache_id}.pkl"
        if cache_path.exists() and use_cache:
            print(f"Loading from cache {cache_path}")
            self.dataset = pickle.load(open(cache_path, "rb"))
        else:
            print(f"{cache_path} not found. Loading from HF {hf_path}/{dataset_name}/{split}")
            hf_dataset = load_dataset(
                path=hf_path, name=dataset_name, split=split, streaming=streaming, **kwargs
            )
            shuffled_dataset = hf_dataset.shuffle(seed=seed)
            # Take n samples that have less than max_tokens_per_doc
            for sample in shuffled_dataset:
                # Get the relevant text item from row
                sample_text = sample[text_field]

                n_tokens = utils.num_tokens_from_messages(
                    messages=[{"role": "user", "content": sample_text}],
                )
                if n_tokens <= max_tokens_per_doc:
                    self.dataset.append(sample_text)
                if len(self.dataset) >= n_samples:
                    break
            assert len(self.dataset) == n_samples
            # Save to cache
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            pickle.dump(self.dataset, open(cache_path, "wb"))


class RandomCharDataset(BaseTextDataset):
    def __init__(self, n_samples, seed=0, lengths=[5, 10, 20, 50, 100]):
        super().__init__(seed)

        self.dataset = []
        # Printable ASCII characters
        ascii_chars = list(string.ascii_letters + string.digits + string.punctuation + " ")
        assert len(ascii_chars) == 95
        for i in range(n_samples):
            length = self.rng.choice(lengths)
            n_char_string = "".join(self.rng.choice(ascii_chars) for _ in range(length))
            self.dataset.append(n_char_string)


class RandomNumberDataset(BaseTextDataset):
    def __init__(self, n_samples, seed=0, lengths=[5, 10, 20, 50, 100]):
        super().__init__(seed)

        self.dataset = []
        for i in range(n_samples):
            length = self.rng.choice(lengths)
            n_digit_string = "".join(
                str(digit) for digit in self.rng.integers(low=0, high=10, size=length)
            )
            self.dataset.append(n_digit_string)


class RandomCharAndNumberDataset(BaseTextDataset):
    def __init__(self, n_samples, seed=0, lengths=[5, 10, 20, 50, 100]):
        super().__init__(seed)

        char_dataset = RandomCharDataset(n_samples // 2, seed, lengths)
        number_dataset = RandomNumberDataset(n_samples - (n_samples // 2), seed, lengths)

        self.dataset = char_dataset.dataset + number_dataset.dataset


class RandomWordsDataset(BaseTextDataset):
    def __init__(self, n_samples, seed=0, lengths=[5, 10, 20, 50, 100]):
        super().__init__(seed)

        nltk.download("words")
        word_list = nltk.corpus.words.words()

        self.dataset = []
        for i in range(n_samples):
            length = self.rng.choice(lengths)
            random_words_string = " ".join(self.rng.choice(word_list) for _ in range(length))
            self.dataset.append(random_words_string)


"""
Task datasets
"""


class BaseTaskDataset:
    def __init__(self, seed=0):
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.dataset = []
        self.type = self.__class__.__name__

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Handling for a slice object
            start = idx.start if idx.start is not None else 0
            stop = idx.stop if idx.stop is not None else len(self.dataset)
            step = idx.step if idx.step is not None else 1
            return [
                {
                    "prompt": self.dataset[i]["prompt"],
                    "output": self.dataset[i]["output"],
                    "type": self.type,
                    "length": len(self.dataset[i]["prompt"]) + len(self.dataset[i]["output"]),
                }
                for i in range(start, stop, step)
            ]
        else:
            # Handling for a plain index
            return {
                "prompt": self.dataset[idx]["prompt"],
                "output": self.dataset[idx]["output"],
                "type": self.type,
                "length": len(self.dataset[idx]["prompt"]) + len(self.dataset[idx]["output"]),
            }


class HFTaskDataset(BaseTaskDataset):
    def __init__(
        self,
        hf_path,
        dataset_name,
        split,
        n_samples,
        prompt_func,
        output_func,
        streaming=True,
        seed=0,
        cache_dir="/tmp/hf_cache",
        max_tokens_per_doc=4096,
        use_cache=False,
    ):
        super().__init__(seed=seed)
        self.type = hf_path

        cache_id = f"{hf_path}_{dataset_name}_{split}_{n_samples}_{streaming}_{seed}"
        cache_path = Path(cache_dir) / f"{cache_id}.pkl"
        if cache_path.exists() and use_cache:
            print(f"Loading from cache {cache_path}")
            self.dataset = pickle.load(open(cache_path, "rb"))
        else:
            print(f"{cache_path} not found. Loading from HF {hf_path}/{dataset_name}/{split}")
            hf_dataset = load_dataset(
                path=hf_path, name=dataset_name, split=split, streaming=streaming
            )
            shuffled_dataset = hf_dataset.shuffle(seed=seed)
            # Take n samples that have less than max_tokens_per_doc
            for sample in shuffled_dataset:
                sample_prompt = str(prompt_func(sample))
                sample_output = str(output_func(sample))
                n_tokens = utils.num_tokens_from_messages(
                    messages=[
                        {"role": "user", "content": sample_prompt},
                        {"role": "user", "content": sample_output},
                    ],
                )
                if n_tokens <= max_tokens_per_doc:
                    self.dataset.append(
                        {
                            "prompt": sample_prompt,
                            "output": sample_output,
                        }
                    )
                if len(self.dataset) >= n_samples:
                    break
            assert len(self.dataset) == n_samples
            # Save to cache
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            pickle.dump(self.dataset, open(cache_path, "wb"))
