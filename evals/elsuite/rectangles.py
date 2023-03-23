import re

import evals
import evals.metrics
import evals.record

task_description = """Give coordinates of 2D axis-aligned rectangles according to some specification, in a single code block with each rectangle on its own line, giving first the range of x coordinates, then the range of y coordinates.

Note that every constraint should be strictly satisfied: for instance, two rectangles that just touch are considered neither disjoint nor overlapping. If the specification is satisfiable then respond with only one code block according to the format given above, and no other text. Otherwise respond with just the word "impossible"."""


def anglicize_constr(constraint):
    if constraint["kind"] == "contains":
        return f"{constraint['big']} contains {constraint['small']}"
    elif constraint["kind"] == "disjoint":
        return f"{' and '.join(constraint['rects'])} are disjoint"
    elif constraint["kind"] == "overlap":
        return f"{' and '.join(constraint['rects'])} overlap"
    elif constraint["kind"] == "uncovered":
        return f"{constraint['cover']} does not fully cover {constraint['peek']}"


def anglicize(sample):
    names = ", ".join(sample["names"])
    constraints = "; ".join(anglicize_constr(constraint) for constraint in sample["constraints"])
    return f"Rectangles {names} where: {constraints}."


float_regex = r"(-?\d+(?:\.\d+)?)"
range_regex = rf"\[\s*{float_regex}\s*,\s*{float_regex}\s*\]"
rect_regex = rf"\(\s*{range_regex}\s*,\s*{range_regex}\s*\)"
regex = re.compile(rf"^\s*(\w+)\s*=\s*{rect_regex}\s*$")


def parse(sampled):
    rectangles = {}
    for line in sampled.splitlines():
        match = re.match(regex, line)
        if match is not None:
            name, x0, x1, y0, y1 = match.groups()
            rectangles[name] = (float(x0), float(x1), float(y0), float(y1))
    return rectangles


def contains(big, small):
    ax0, ax1, ay0, ay1 = big
    bx0, bx1, by0, by1 = small
    return ax0 < bx0 and bx1 < ax1 and ay0 < by0 and by1 < ay1


def disjoint(rects):
    (ax0, ax1, ay0, ay1), (bx0, bx1, by0, by1) = rects
    return ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0


def overlap(rects):
    (ax0, ax1, ay0, ay1), (bx0, bx1, by0, by1) = rects
    return not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0)


def uncovered(cover, peek):
    (ax0, ax1, ay0, ay1), (bx0, bx1, by0, by1) = cover, peek
    return not (ax0 <= bx0 and bx1 <= ax1 and ay0 <= by0 and by1 <= ay1)


def is_match(constraint, rectangles):
    if constraint["kind"] == "contains":
        return contains(rectangles[constraint["big"]], rectangles[constraint["small"]])
    elif constraint["kind"] == "disjoint":
        return disjoint([rectangles[name] for name in constraint["rects"]])
    elif constraint["kind"] == "overlap":
        return overlap([rectangles[name] for name in constraint["rects"]])
    elif constraint["kind"] == "uncovered":
        return uncovered(rectangles[constraint["cover"]], rectangles[constraint["peek"]])


class Rectangles(evals.Eval):
    def __init__(self, train_jsonl, test_jsonl, **kwargs):
        super().__init__(**kwargs)
        self.train_jsonl = train_jsonl
        self.test_jsonl = test_jsonl

    def eval_sample(self, sample, rng):
        sat_example = rng.choice(self.sat_examples)
        unsat_example = rng.choice(self.unsat_examples)

        prompt = [
            {"role": "system", "content": task_description},
            {
                "role": "system",
                "content": anglicize(sat_example),
                "name": "example_user",
            },
            {
                "role": "system",
                "content": sat_example["example"],
                "name": "example_assistant",
            },
            {
                "role": "system",
                "content": anglicize(unsat_example),
                "name": "example_user",
            },
            {"role": "system", "content": "impossible", "name": "example_assistant"},
            {"role": "user", "content": anglicize(sample)},
        ]

        result, actual_prompt, metadata = evals.completion_query(self.model_spec, prompt)
        sampled = result["choices"][0]["text"]
        evals.record.record_sampling(prompt=actual_prompt, sampled=sampled, metadata=metadata)
        if sample["satisfiable"]:
            rectangles = parse(sampled)
            correct = (
                set(rectangles.keys()) == set(sample["names"])
                and all(x0 < x1 and y0 < y1 for x0, x1, y0, y1 in rectangles.values())
                and all(is_match(constraint, rectangles) for constraint in sample["constraints"])
            )
        else:
            correct = "impossible" in sampled or "Impossible" in sampled
        evals.record.record_match(correct, sampled=sampled)

    def run(self, recorder):
        train_samples = evals.get_jsonl(self.train_jsonl)
        self.sat_examples = [x for x in train_samples if x["satisfiable"]]
        self.unsat_examples = [x for x in train_samples if not x["satisfiable"]]
        self.eval_all_samples(recorder, evals.get_jsonl(self.test_jsonl))
        return {"accuracy": evals.metrics.get_accuracy(recorder.get_events("match"))}
