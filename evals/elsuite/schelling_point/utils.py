import json
import logging
from typing import Any, Callable, Tuple
from evals.registry import is_chat_model
from evals.elsuite.schelling_point.prompts import hhh_prompt
import evals

def replace_last(s: str, old: str, new: str) -> str:
    # Reverse the string, replace the first occurrence, then reverse it back
    return s[::-1].replace(old[::-1], new[::-1], 1)[::-1]


def get_response(
    completion_fn: Callable[..., Any], sys_prompt: str, user_prompt: str, temperature: float
) -> Tuple[str, str]:
    """
    Takes completion_fn and wraps sys_prompt and user_prompt appropriately
    depending on whether the model is a chat model or not. Also parses the
    response via json and returns the output and scratchpad.
    """

    if is_chat_model(completion_fn.model):
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = completion_fn(messages, temperature=temperature).get_completions()[0]
    else:

        prompt = f"{hhh_prompt}System: {sys_prompt}\nHuman: {user_prompt}\n\nAssistant: {{"
        response = (
            "{"
            + completion_fn(prompt, max_tokens=250, temperature=temperature).get_completions()[0]
        )

        # cut text off after and including 'User:'
        response = response.split("Human:")[0]

        # cut text off after and including 'System:'
        response = response.split("System:")[0]

        # return the first complete '{' '}' pair
        start_pair = response.find("{")
        
        end_pair = response.find("}")

        if start_pair == -1 or end_pair == -1 or start_pair > end_pair:
            return response, "error"

        response = response[start_pair : end_pair + 1]

    # replace “ ” with " "
    response = response.replace("“", '"').replace("”", '"')

    # replace all quotes with escaped double quotes
    response = response.replace("'", '"').replace('"', '\\"')

    # fix the escaped double quotes outside "scratchpad" and "output"
    response = response.replace('\\"scratchpad\\"', '"scratchpad"').replace(
        '\\"output\\"', '"output"'
    )

    # fix the escaped double quotes that start and end the value fields
    response = (
        response.replace(': \\"', ': "')
        .replace('\\"}', '"}')
        .replace('\\"\n', '"\n')
        .replace('\\" }', '" }')
    )
    response = replace_last(response, '\\",', '",')

    try:
        response = json.loads(response)
        if type(response) == str:
            # format is typically "'scratchpad': ..., 'output': ..."
            scratchpad = response.split("'scratchpad':")[1].split("'output':")[0].strip()
            output = response.split("'output':")[1].strip()
        else:
            output = str(response["output"]).lower().strip()
            scratchpad = response["scratchpad"].lower().strip()

        return output, scratchpad

    except Exception:

        logging.warn(f"ERROR: incorrect json parsing. Model output: {response}")

        evals.record.record_metrics(
            is_runtime_error=True,
        )

        # special case for random_numbers dataset
        if type(response) == int:
            return str(response), "error"

        if type(response) == dict:
            return "error", "error"

        return response.lower().strip(), "error"

