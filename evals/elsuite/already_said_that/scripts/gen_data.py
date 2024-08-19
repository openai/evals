import argparse
import os
import random
import json

import nltk
from nltk.corpus import wordnet
from tqdm.auto import tqdm


def process_wordnet() -> list[str]:
    """
    Process the wordnet corpus and save it to the given directory
    License info: https://www.nltk.org/nltk_data (number 102)
    """
    # download wordnet corpus if necessary
    nltk.download("wordnet", force=True)
    wordnet_words = wordnet.words()
    # get all unique alpha words from wordnet corpus
    words = set()
    for word in tqdm(wordnet_words):
        if word.isalpha():
            words.add(word.lower())

    return list(words)


def gen_sample(words_corpus: list[str], n_words, rng: random.Random) -> dict:
    words = rng.sample(words_corpus, n_words)
    return {"words": words}


def gen_samples(n_samples: int, n_words: int, rng: random.Random) -> list[dict]:
    words = process_wordnet()
    samples = []
    for _ in tqdm(range(n_samples)):
        sample = gen_sample(words, n_words, rng)
        samples.append(sample)
    return samples


def write_to_jsonl(
    samples: list[dict],
    jsonl_path: str,
):
    with open(jsonl_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")


def main(args: argparse.Namespace):
    rng = random.Random(args.seed)
    samples = gen_samples(args.n_samples, args.n_words, rng)
    os.makedirs(args.jsonl_dir, exist_ok=True)
    jsonl_path = os.path.join(args.jsonl_dir, f"{args.n_samples}_{args.n_words}.jsonl")
    write_to_jsonl(samples, jsonl_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument(
        "--n_words", type=int, default=100, help="Number of words in each sample"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--jsonl_dir", type=str, default="./evals/registry/data/already_said_that/"
    )

    args = parser.parse_args()

    main(args)
