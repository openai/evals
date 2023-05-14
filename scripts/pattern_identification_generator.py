import json
import random

SYMBOLS = list("abcdefghijklmnopqrstuvwxyz")
DELIMETER = "->"
INSTRUCTION = "Figure out the pattern in the below examples, and then answer with just \"foo\" or \"bar\"."
TASK_NAME = "pattern_identification"
random.seed(42)


def generate_example():
    num_symbols = int(len(SYMBOLS) / 2)
    target_symbol = random.choice(SYMBOLS)
    symbol_list = random.sample(SYMBOLS, num_symbols)
    target = "foo" if target_symbol in symbol_list else "bar"
    return target_symbol, symbol_list, target


def generate_exemplars_str(num_exemplars: int = 8):
    exemplars = [generate_example() for _ in range(num_exemplars)]
    return "\n".join([INSTRUCTION] + [f"({ex[0]}, {ex[1]}) {DELIMETER} {ex[2]}".replace("'", "") for ex in exemplars])


def generate_eval_examples(num_eval_examples: int = 250):
    return [generate_example() for _ in range(num_eval_examples)]


if __name__ == "__main__":
    eval_examples = generate_eval_examples()
    output_path = f"evals/registry/data/{TASK_NAME}/samples.v0.jsonl"
    with open(output_path, "w") as writer:
        for example in eval_examples:
            d = {
                "input": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"{generate_exemplars_str()}\n({example[0]}, {example[1]}) {DELIMETER}".replace("'", "")},
                ],
                "ideal": example[2],
            }
            writer.write(json.dumps(d) + "\n")
    print(f"{len(eval_examples)} lines written to {output_path}.")
