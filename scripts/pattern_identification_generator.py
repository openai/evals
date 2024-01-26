"""
    Usage: python scripts/task_identification_generator.py
    This script generates examples for a task identification challenge, 
    with the task being to identify patterns between a set of symbols and their resulting labels.
"""

import json
import os
import random
from typing import Literal

# Ensure consistent results across runs
random.seed(42)

SYMBOLS = list("abcdefghijklmnopqrstuvwxyz")
DELIMETER = "->"
INSTRUCTION = (
    'Figure out the pattern in the below examples, and then answer with just "foo" or "bar".'
)
TASK_NAME = "pattern_identification"

# This function generates an example symbol set and its corresponding label
def generate_example() -> tuple[str, list[str], Literal["foo", "bar"]]:
    num_symbols = int(len(SYMBOLS) / 2)
    target_symbol = random.choice(SYMBOLS)
    symbol_list = random.sample(SYMBOLS, num_symbols)
    target: Literal["foo", "bar"] = "foo" if target_symbol in symbol_list else "bar"
    return (target_symbol, symbol_list, target)


# This function generates a string of multiple examples, used to give a user multiple attempts to identify the pattern
def generate_exemplars_str(num_exemplars: int = 8) -> str:
    exemplars = [generate_example() for _ in range(num_exemplars)]
    exemplars_str = [
        f"({exemplar[0]}, {exemplar[1]}) {DELIMETER} {exemplar[2]}".replace("'", "")
        for exemplar in exemplars
    ]
    return "\n".join([INSTRUCTION] + exemplars_str)


# This function generates a set of evaluation examples and their corresponding labels
def generate_eval_examples(
    num_eval_examples: int = 250,
) -> tuple[list[str], list[Literal["foo", "bar"]]]:
    eval_examples = [generate_example() for _ in range(num_eval_examples)]
    eval_examples_str = [
        f"{generate_exemplars_str()}\n({example[0]}, {example[1]}) {DELIMETER}".replace("'", "")
        for example in eval_examples
    ]
    targets: list[Literal["foo", "bar"]] = [example[2] for example in eval_examples]
    return eval_examples_str, targets


if __name__ == "__main__":
    eval_examples_str, targets = generate_eval_examples()

    # Generate the output path in a OS-agnostic manner
    output_path = os.path.join("evals", "registry", "data", TASK_NAME, "samples.v0.jsonl")

    with open(output_path, "w") as writer:
        for eval_example_str, target in zip(eval_examples_str, targets):
            d = {
                "input": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": eval_example_str},
                ],
                "ideal": target,
            }
            writer.write(json.dumps(d) + "\n")
    print(f"{len(eval_examples_str)} lines written to {output_path}.")
