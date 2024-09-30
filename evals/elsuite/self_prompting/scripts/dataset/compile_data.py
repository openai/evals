import json

import numpy as np
from eval_list import eval_list

import evals.data
from evals.registry import Registry

np.random.seed(42)
min_samples_per_dataset = 50
n_test_samples = 10

seen = set()
datarows = []
for eval in Registry().get_evals("*"):
    if eval.key not in eval_list or eval.key in seen:
        continue
    seen.add(eval.key)

    if eval.args and "samples_jsonl" in eval.args:

        samples = evals.data.get_jsonl(eval.args["samples_jsonl"])

        # Contruct our tasks dataset
        instruction_input_output = []
        for sample in samples:
            if "input" in sample and "ideal" in sample:
                # We only want single-system single-user samples:
                if isinstance(sample["input"], list) and len(sample["input"]) == 2:
                    if (
                        sample["input"][0]["role"] == "system"
                        and sample["input"][1]["role"] == "user"
                    ):
                        # Skip if output is a list
                        if isinstance(sample["ideal"], list):
                            continue

                        dp_instruction = sample["input"][0]["content"]
                        dp_in = sample["input"][1]["content"]
                        dp_out = sample["ideal"]

                        instruction_input_output.append((dp_instruction, dp_in, dp_out))

        # Skip if there are not enough samples
        if len(instruction_input_output) < min_samples_per_dataset:
            continue
        # Check that all dp_instruction are the same
        instruction_input_output = sorted(instruction_input_output, key=lambda x: x[0])
        if instruction_input_output[0][0] != instruction_input_output[-1][0]:
            continue

        # Shuffle samples
        np.random.shuffle(instruction_input_output)

        test_samples = [
            {
                "input": i,
                "output": o,
            }
            for _, i, o in instruction_input_output[:n_test_samples]
        ]
        train_samples = [
            {
                "input": i,
                "output": o,
            }
            for _, i, o in instruction_input_output[n_test_samples:]
        ]

        row = {
            "eval": eval.key,
            "instruction": instruction_input_output[0][0],
            "test_samples": test_samples,
            "train_samples": train_samples,
        }
        datarows.append(row)

assert len(datarows) == len(
    eval_list
), f"Unexpected number of evals: {len(datarows)} != {len(eval_list)}"
assert set([r["eval"] for r in datarows]) == set(
    eval_list
), f"Missing evals: {set(eval_list) - set([r['eval'] for r in datarows])}"

# Shuffle rows
np.random.shuffle(datarows)

# Save jsonl to file
with open("samples.jsonl", "w") as f:
    for row in datarows:
        f.write(json.dumps(row) + "\n")
