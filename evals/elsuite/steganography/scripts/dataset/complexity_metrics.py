import gzip
from collections import Counter

import numpy as np
from scipy.stats import entropy


def calculate_entropy(text):
    unique_chars, counts = np.unique(list(text), return_counts=True)
    probabilities = counts / len(text)
    return entropy(probabilities, base=2)


def calculate_compression_ratio(text):
    text_bytes = text.encode("utf-8")
    compressed = gzip.compress(text_bytes)
    return len(compressed) / len(text_bytes)


def calculate_brevity_score(text):
    words = text.split()
    total_words = len(words)
    unique_words = len(set(words))
    word_counts = Counter(words)
    frequencies = [word_counts[word] / total_words for word in set(words)]

    return sum(frequencies) / unique_words


if __name__ == "__main__":
    text = "Example text to calculate entropy."
    entropy_value = calculate_entropy(text)
    print(entropy_value)

    text = "Example text to calculate compression ratio."
    ratio = calculate_compression_ratio(text)
    print(ratio)

    text = "Example text to calculate the brevity score."
    score = calculate_brevity_score(text)
    print(score)
