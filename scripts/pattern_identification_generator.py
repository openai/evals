"""
    Usage: python scripts/task_identification_generator.py
"""

import json
import random
from typing import Literal

random.seed(42)

SYMBOLS = list("abcdefghijklmnopqrstuvwxyz")
DELIMETER = "->"
INSTRUCTION = (
    'Figure out the pattern in the below examples, and then answer with just "foo" or "bar".'
)
TASK_NAME = "pattern_identification"


def generate_example() -> tuple[str, list[str], Literal["foo", "bar"]]:
    num_symbols = int(len(SYMBOLS) / 2)
    target_symbol = random.choice(SYMBOLS)
    symbol_list = random.sample(SYMBOLS, num_symbols)
    target: Literal["foo", "bar"] = "foo" if target_symbol in symbol_list else "bar"
    return (target_symbol, symbol_list, target)


def generate_exemplars_str(num_exemplars: int = 8) -> str:
    exemplars = [generate_example() for _ in range(num_exemplars)]
    exemplars_str = [
        f"({exemplar[0]}, {exemplar[1]}) {DELIMETER} {exemplar[2]}".replace("'", "")
        for exemplar in exemplars
    ]
    return "\n".join([INSTRUCTION] + exemplars_str)


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
    output_path = f"evals/registry/data/{TASK_NAME}/samples.v0.jsonl"
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
