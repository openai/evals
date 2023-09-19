from typing import Optional

import tiktoken

from evals.elsuite.ballots.prompts import (
    control_chat_prompt,
    control_text_template,
    manipulation_chat_template,
    manipulation_text_template,
    text_prompt,
    voter_chat_prompt,
    voter_text_prompt,
)
from evals.registry import is_chat_model

LOGIT_BIAS_MAX = 100
LOGIT_BIAS_MIN = -100
VOTER_DECISION_TOKS = [" Yes", " No"]
BASE_REMOVED_TOKS = [" User", " Assistant", " AI"]
YES_DIRECTION = "yes"
NO_DIRECTION = "no"
CONTROL_DIRECTION = "control"
DOUBLE_NEWLINE_TOKS = [
    "\n\n",
]
ALL_DOUBLE_NEWLINE_TOKS = DOUBLE_NEWLINE_TOKS + [
    ".\n\n",
    ";\n\n",
    " }\n\n",
    ");\n\n",
    "}\n\n",
    ")\n\n",
    "\n\n\n\n",
    ">\n\n",
    "\n\n\n",
    "();\n\n",
    ":\n\n",
    " {\n\n",
    '"\n\n',
    "?\n\n",
    ".\n\n\n\n",
    "...\n\n",
    '."\n\n',
    "!\n\n",
    "};\n\n",
    "';\n\n",
    "]\n\n",
    " */\n\n",
    "()\n\n",
    ".\u201d\n\n",
    " });\n\n",
    '");\n\n',
    "));\n\n",
    '";\n\n',
    "\u3002\n\n",
]


def toks_to_id(model: str, toks: list[str], validation: Optional[list[str]] = None) -> list[str]:
    if model == "dummy":
        model = "gpt-3.5-turbo"  # pretend it's a chat model

    encoder = tiktoken.encoding_for_model(model)
    toks = [encoder.encode(tok) for tok in toks]
    assert all(len(tok) == 1 for tok in toks), f"Invalid toks {toks}"
    toks = [str(tok[0]) for tok in toks]
    if validation:
        assert sorted(toks) == sorted(validation), f"Lists are different {toks} {validation}"
    return toks


def format_messages(messages, *format_args, **format_kwargs):
    # Format the content of all the messages
    return [
        {
            "role": message["role"],
            "content": message["content"].format(*format_args, **format_kwargs),
        }
        for message in messages
    ]


def format_prompt(prompt, *format_args, **format_kwargs):
    # Format the content of all the messages if prompt is a list of messages, otherwise format the prompt string
    if isinstance(prompt, str):
        return prompt.format(*format_args, **format_kwargs)
    elif isinstance(prompt, list):
        return format_messages(prompt, *format_args, **format_kwargs)


def get_influencer_prompt(model, direction):
    yes_chat_prompt = format_messages(manipulation_chat_template, condition="Yes")
    no_chat_prompt = format_messages(manipulation_chat_template, condition="No")

    yes_text_prompt = (
        text_prompt + manipulation_text_template.format(condition="Yes") + "\n\n---\n\n"
    )
    no_text_prompt = text_prompt + manipulation_text_template.format(condition="No") + "\n\n---\n\n"
    control_text_prompt = text_prompt + control_text_template + "\n\n---\n\n"

    if is_chat_model(model):
        return {
            YES_DIRECTION: yes_chat_prompt,
            NO_DIRECTION: no_chat_prompt,
            CONTROL_DIRECTION: control_chat_prompt,
        }[direction.lower()]
    else:
        return {
            YES_DIRECTION: yes_text_prompt,
            NO_DIRECTION: no_text_prompt,
            CONTROL_DIRECTION: control_text_prompt,
        }[direction.lower()]


def get_voter_prompt(model):
    if is_chat_model(model):
        return voter_chat_prompt
    else:
        return voter_text_prompt


def prompt_matches_model(model, prompt):
    if is_chat_model(model):
        return isinstance(prompt, list)
    else:
        return isinstance(prompt, str)


def reverse_roles(messages):
    return [
        {
            "role": "user" if message["role"] == "assistant" else "assistant",
            "content": message["content"],
        }
        for message in messages
    ]


def chat_to_text(messages):
    return "\n".join(
        [f"{message['role'].capitalize()}: {message['content']}" for message in messages]
    )
