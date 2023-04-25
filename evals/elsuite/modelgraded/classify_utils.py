import logging
import string
from typing import Callable, Iterable, Optional, Union

from evals.elsuite.utils import format_necessary, format_prompt
from evals.prompt.base import OpenAICreateChatPrompt, is_chat_prompt

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


def get_choice(
    text: str, eval_type: str, match_fn: Union[str, Callable], choice_strings: Iterable[str]
) -> str:
    """Clean the answer string to a choice string to one of choice_strings. Return '__invalid__.' if no match."""
    if isinstance(match_fn, str):
        match_fn = MATCH_FNS[match_fn]
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
    logging.warn(f"Choices {choice_strings} not parsable for {eval_type}: {text}")
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


def format_classify(
    prompt: OpenAICreateChatPrompt,
    format_type: str = "in_message",
    input_outputs: Optional[dict[str, str]] = None,
    **format_kwargs: dict[str, OpenAICreateChatPrompt],
) -> OpenAICreateChatPrompt:
    """Return an OpenAICreateChatPrompt that can be passed PromptFn for modelgraded eval.

    'in_message' returns: [
        {
            "role": "user",
            "content": \"""
                User: {input}
                Assistant: {completion}

                Was the assistant response helpful?
                \""".strip(),
        }
    ]

    'out_message' returns: [
        {"role": "user", "content": "{input}"},
        {"role": "assistant", "content": "{completion}"},
        {"role": "user", "content": "Was the last assistant response helpful?"},
    ]
    """
    input_outputs = input_outputs or {}
    if format_type == "in_message":
        return format_prompt(prompt, **format_kwargs)
    elif format_type == "out_message":
        assert len(input_outputs) == 1, "out_message only supports one input/output pair"
        # extra input-output data, as it is treated specially
        input_completions = {
            k: (k, format_kwargs[k], v, format_kwargs[v]) for k, v in input_outputs.items()
        }
        format_kwargs = {
            k: v
            for k, v in format_kwargs.items()
            if k not in input_outputs.values() and k not in input_outputs
        }
        convo = []
        for input_key, input, completion_key, completion in input_completions.values():
            del input_key, completion_key
            assert isinstance(completion, str), f"completion must be str, not {type(completion)}"
            if is_chat_prompt(input):
                convo += input
            else:
                convo.append({"role": "user", "content": input})
            convo.append({"role": "assistant", "content": completion})
        return convo + format_prompt(prompt, **format_kwargs)
    else:
        raise ValueError(f"format_type must be 'in_message' or 'out_message', not {format_type}")
