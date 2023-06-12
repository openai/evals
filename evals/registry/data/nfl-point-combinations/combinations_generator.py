# generate a jsonl where each line is a sample input

import json
import os

DATA_PATH = os.path.dirname(__file__)

file_name = f"{DATA_PATH}/samples.jsonl"


def ways_to_score(n):
    scores = [2, 3, 6, 7]
    dp = [0] * (n + 1)
    dp[0] = 1

    for score in scores:
        for i in range(score, n + 1):
            dp[i] += dp[i - score]

    return dp[n]


samples = []

for i in range(1, 41):
    answer = ways_to_score(i)

    prompt = f"As of the year 2010, in American Football, how many unique, order-independent ways can an NFL (National Football League) team score exactly {i} points in a single game? Exclude one-point safeties as one of the scoring options. List out all the possible combinations and write your final answer as a single number enclosed in square brackets."

    samples.append(
        {
            "input": [{"role": "system", "content": prompt}],
            "ideal": str(f"[{answer}]"),
        }
    )

    # save samples jsonl
    with open(file_name, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
