import argparse
import dis
import json
import math

DEFAULT_RANGE = [0, 100]  # inclusive


def get_func_from_code(code):
    return lambda x: eval(code, {"math": math, "x": x})


def get_complexity(code: str) -> int:
    #   NOTE: this is quite ugly, but should be good enough for dataset-creating code
    code = "global func_name\ndef func_name(x): return " + code
    exec(code)
    return len(list(dis.get_instructions(func_name)))


def create_dataset(out_file, in_file):
    samples = []

    for line in in_file:
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        func = get_func_from_code(line)
        values = list(int(func(x)) for x in range(DEFAULT_RANGE[0], DEFAULT_RANGE[1] + 1))
        samples.append(
            {
                "code": line,
                "complexity": get_complexity(line),
                "range": DEFAULT_RANGE,
                "values": values,
            }
        )

    #   Ensure we don't have duplicates - they might be different functions, but if they return the same
    #   value for every input in the DEFAULT_RANGE then they are in fact the same sample.
    for sample_ix, sample in enumerate(samples):
        for other_sample in samples[sample_ix + 1 :]:
            if sample["values"] == other_sample["values"]:
                raise ValueError(
                    f"Samples {sample['code']} and {other_sample['code']} are indistinguishable"
                )

    samples.sort(key=lambda x: x["complexity"])

    for i, sample in enumerate(samples):
        sample = dict(sample_ix=i, **sample)
        json.dump(sample, out_file)
        out_file.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=argparse.FileType("w"), required=True)
    parser.add_argument("--in", dest="in_", type=argparse.FileType("r"), default="raw_code.txt")
    args = parser.parse_args()
    create_dataset(out_file=args.out, in_file=args.in_)
