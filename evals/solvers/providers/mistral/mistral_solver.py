from typing import Any, Dict, List

from evals.solvers.providers.openai.third_party_solver import ThirdPartySolver


def _sanitize_tool_call_id(tool_call_id: str):
    """Mistral expects tool_call_ids to be 9 alphanumeric characters."""
    return tool_call_id.replace("_", "")[:9]


class MistralSolver(ThirdPartySolver):
    def __init__(self, **kwargs):
        super().__init__(
            api_base="https://api.mistral.ai/v1", api_key_env_var="MISTRAL_API_KEY", **kwargs
        )

    def _process_msgs(self, messages: List[Dict[str, Any]]):
        """
        Mistral expects the assistant text response and tool calls in separate messages,
        so we split the assistant message if necessary.
        """
        out_messages = []
        for msg in messages:
            if msg["role"] == "assistant" and msg.get("tool_calls"):
                tool_calls = msg["tool_calls"]
                for tool_call in tool_calls:
                    tool_call["id"] = _sanitize_tool_call_id(tool_call["id"])
                if msg["content"] != "":
                    text_msg = msg.copy()
                    text_msg.pop("tool_calls")
                    out_messages.append(text_msg)
                    msg["content"] = ""
            elif msg["role"] == "tool" and msg.get("tool_call_id"):
                msg["tool_call_id"] = _sanitize_tool_call_id(msg["tool_call_id"])
            out_messages.append(msg)
        return out_messages
