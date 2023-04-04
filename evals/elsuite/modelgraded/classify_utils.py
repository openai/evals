import itertools
import string
from typing import Callable, Iterable

from evals.elsuite.utils import format_necessary

INVALID_STR = "__invalid__"
CHOICE_KEY = "choice"


ANSWER_PROMPTS = {
    # e.g. "Yes"
    "classify": "Answer the question by printing only a single choice from {choices} (without quotes or punctuation) corresponding to the correct answer with no other text.".strip(),
    # e.g. "Yes\n The reasons are: ..."
    "classify_cot": "First, answer by printing a single choice from {choices} (without quotes or punctuation) corresponding to the correct answer. Then, from the next line, explain your reasonings step by step.".strip(),
    # e.g. "Let's think step by step. ...\nYes"
    "cot_classify": """
First, write out in a step by step manner your reasoning to be sure that your conclusion is correct. Avoid simply stating the correct answer at the outset. Then print only a single choice from {choices} (without quotes or punctuation) on its own line corresponding to the correct answer. At the end, repeat just the answer by itself on a new line.

Reasoning:""".strip(),
    "cot_classify_jp": """
まず、一歩一歩あなたの推論を書き出してください。単に正しい答えを最初に述べることを避けてください。次に、{choices}（引用符や句読点なし）から正しい答えに対応する1つの選択肢を単独の行に書きだしてください。最後に、答えだけを新しい行に繰り返してください。

推論：
    """.strip(),
}
MATCH_FNS = {
    "include": lambda x, y: float(x in y),
    "exact": lambda x, y: float(x == y),
    "endswith": lambda x, y: x.endswith(y),
    "starts_or_endswith": lambda x, y: x.startswith(y) or x.endswith(y),
}


def choice_to_str(choice_strings: Iterable[str]) -> str:
    """Return a string of choices, e.g. '"Yes" or "No" or "Maybe"'."""
    return " or ".join(f'"{choice}"' for choice in choice_strings)


def get_choice(text: str, eval_type: str, match_fn: Callable, choice_strings: Iterable[str]) -> str:
    """Clean the answer string to a choice string to one of choice_strings. Return '__invalid__.' if no match."""
    lines = text.strip().split("\n")
    if eval_type.startswith("cot_classify"):
        lines = lines[::-1]  # reverse lines
    for line in lines:
        line = line.strip()
        line = "".join(c for c in line if c not in string.punctuation)
        if not line:
            continue
        for choice in choice_strings:
            if match_fn(line, choice):
                return choice
    return INVALID_STR


def concat_n_completions(completions: Iterable[str], template_i: str) -> str:
    """Concatenate n completions into a single text string."""
    completion = ""
    for i, completion_i in enumerate(completions):
        completion += format_necessary(
            template_i,
            i=i + 1,
            i_abc=string.ascii_lowercase[i % 26],
            i_ABC=string.ascii_uppercase[i % 26],
            output=completion_i,
            n=len(completions),
        )
    return completion.strip()


def expand_args_dict(args_dict):
    """Expand a dict of dicts, with namings.

    args_dict = {
        "a": {"a1": 1, "a2": 2},
        "b": {"b1": 3, "b2": 4},
    }
    expand_args_dict(args_dict) = {
        "a=a1:b=b1": {"a": ("a1", 1), "b": ("b1", 3)},
        "a=a1:b=b2": {"a": ("a1", 1), "b": ("b2", 4)},
    ...}
    """
    if not args_dict:
        return {}
    args_dict = {k: list(v.items()) for k, v in args_dict.items()}
    keys = list(args_dict.keys())
    values = list(args_dict.values())
    new_values = [dict(zip(keys, v)) for v in itertools.product(*values)]
    new_names = [":".join([f"{k}={v[0]}" for k, v in sorted(d.items())]) for d in new_values]
    return dict(zip(new_names, new_values))
