import json

import numpy as np


def gen_fourier_series(n=4):
    terms = []
    for i in range(n):
        A = np.random.randint(1, 10)
        B = np.random.randint(1, 10)
        C = np.random.randint(1, 10)
        terms.append((A, B, C))
    return terms


def fourier_series_ans(terms, x=0):
    ans = 0
    for A, B, C in terms:
        ans += A * np.sin(B * x + C)
    return ans


def create_sample(terms, ans):
    return {
        "input": [
            {
                "role": "system",
                "content": "You are Fourier GPT. You will be given a sequence of sine terms. Calculate the output of these sine terms at x = 0.",
            },
            {
                "role": "user",
                "content": "Calculate the output of these sine terms at x = 0: "
                + " ".join([f"{A}*sin({B}*x + {C})" for A, B, C in terms]),
            },
        ],
        "ideal": [str(ans)],
    }


num_samples = 100  # Change this value to generate as many samples as you want

with open("fourier_samples.jsonl", "w") as f:
    for i in range(num_samples):
        terms = gen_fourier_series()
        ans = fourier_series_ans(terms)
        sample = create_sample(terms, ans)
        f.write(json.dumps(sample) + "\n")
