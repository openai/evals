import json
import random
import sys
from io import StringIO

import requests
from word_search_generator import WordSearch

word_site = "https://www.mit.edu/~ecprice/wordlist.10000"
response = requests.get(word_site)
WORDS = response.content.splitlines()
valid_words = [word.decode("ascii") for word in list(WORDS) if len(word) >= 4]


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


# difficulty 1: 1-shot word search with words given
# difficulty 2: 1-shot word search with words not given
# difficulty 3: 0-shot word search with words given
# difficulty 4: 0-shot word search with words not given

system_prefix = {
    "role": "system",
    "content": "You are a helpful assistant. Follow the user's examples and instructions to solve problems.",
}


def gen_dif_1_example(output, example):
    # 1 shot word search with words given
    template = {
        "input": [system_prefix],
        "ideal": output[-1][len("Answer Key: ") :].replace("), ", ").").split("."),
    }
    output = example + output[:-1]
    output.append("Answer Key:")
    template["input"].append({"role": "user", "content": "\n".join(output)})
    return template


def gen_dif_2_example(output, example):
    # 1 shot word search with words not given
    # find the line that starts with Find these words:
    for i, line in enumerate(output):
        if line.startswith("Find these words:"):
            output[i] = "Find all notable words in the above word search."
    for i, line in enumerate(example):
        if line.startswith("Find these words:"):
            example[i] = "Find all notable words in the above word search."
    template = {
        "input": [system_prefix],
        "ideal": output[-1][len("Answer Key: ") :].replace("), ", ").").split("."),
    }
    output = example + output[:-1]
    output.append("Answer Key:")
    template["input"].append({"role": "user", "content": "\n".join(output)})
    return template


def gen_dif_3_example(output, example):
    # 0 shot word search with words given
    # find the line that starts with Find these words:
    template = {
        "input": [system_prefix],
        "ideal": output[-1][len("Answer Key: ") :].replace("), ", ").").split("."),
    }
    output = output[:-1]
    output.append(
        "Answer according to the following format, where coordinates are in the form (1-indexed col # from left, 1-indexed row # from top):"
    )
    output.append(example[-1])  # just the answer key
    output.append("Answer Key:")
    template["input"].append({"role": "user", "content": "\n".join(output)})
    return template


def gen_dif_4_example(output, example):
    # 0 shot word search with words not given
    for i, line in enumerate(output):
        if line.startswith("Find these words:"):
            output[i] = "Find all notable words in the above word search."
    template = {
        "input": [system_prefix],
        "ideal": output[-1][len("Answer Key: ") :].replace("), ", ").").split("."),
    }
    output = output[:-1]
    output.append(
        "Answer according to the following format, where coordinates are in the form (1-indexed col # from left, 1-indexed row # from top):"
    )
    output.append(example[-1])  # just the answer key
    output.append("Answer Key:")
    template["input"].append({"role": "user", "content": "\n".join(output)})
    return template


def generate_word_search(exclude):
    word_sample = random.sample(list(set(valid_words) - set(exclude)), 5)
    puzzle = WordSearch(",".join(word_sample))
    puzzle.directions = "N, NE, E, SE, S, SW, W, NW"
    with Capturing() as output:
        puzzle.show()
    return output, word_sample


def generate(difficulty=1):
    output, words = generate_word_search([])
    example, _ = generate_word_search(words)
    if difficulty == 1:
        return gen_dif_1_example(output, example)
    elif difficulty == 2:
        return gen_dif_2_example(output, example)
    elif difficulty == 3:
        return gen_dif_3_example(output, example)
    elif difficulty == 4:
        return gen_dif_4_example(output, example)
    else:
        raise ValueError("Difficulty must be 1, 2, 3, or 4.")


# generate jsonl file
with open("word_search.jsonl", "w") as f:
    for i in range(100):
        f.write(json.dumps(generate(difficulty=1)) + "\n")
    for i in range(100):
        f.write(json.dumps(generate(difficulty=2)) + "\n")
    for i in range(100):
        f.write(json.dumps(generate(difficulty=3)) + "\n")
    for i in range(100):
        f.write(json.dumps(generate(difficulty=4)) + "\n")
