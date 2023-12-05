import os
import re
from typing import Literal

from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def is_system_msg(m: dict) -> bool:
    assert isinstance(m, dict), "Message must be a dict."
    assert "role" in m, "Message must have a role."
    assert isinstance(m["role"], str), "Message role must be a string."

    return m["role"].endswith("-system")


def format_msg(msg: dict, role: Literal["mark", "conartist"]) -> dict:
    """Formats a single message from the perspective of the `role`."""
    # pre-conditions
    if "system" in msg["role"]:
        assert role in msg["role"], "Can't format system message for other user!"

    # body
    is_others_msg = role not in msg["role"]
    new_content = msg["content"]

    if is_others_msg:
        new_role = "user"
    elif is_system_msg(msg):
        new_role = "system"
    else:
        new_role = "assistant"

    new_message = {
        "content": new_content,
        "role": new_role,
    }

    # post-conditions
    assert isinstance(new_message["content"], str)
    assert new_message["role"] in ["user", "assistant", "system"]

    return new_message


def format_msgs(
    messages: list[dict],
    role: Literal["mark", "conartist"],
) -> list[dict]:
    """Format messages from the perspective of the `role`."""
    new_messages = [format_msg(msg, role) for msg in messages]

    # post-conditions
    for m in new_messages:
        assert m["role"] in ["user", "assistant", "system"]

    return new_messages


def model_broke_character(message: str) -> bool:
    return (
        "AI language model" in message.split("[OPEN]")[1]
        if "[OPEN]" in message
        else "AI language model" in message
    )


def model_output_empty_tags(message: str) -> bool:
    pattern = r"\[OPEN\]\s*$"
    return bool(re.search(pattern, message))


def openai_chatcompletion_create(*args, **kwargs):
    return client.chat.completions.create(*args, **kwargs)
