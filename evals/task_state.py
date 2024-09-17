from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Message:
    """
    A single message in a conversation.

    Args:
        role: Typically "system", "user", or "assistant" but can also take other
            values depending on the task (e.g. "player1", "player2").
        content: The string content of the message.
    """

    role: str
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

    def to_dict(self):
        d = asdict(self)
        if self.tool_calls is None:
            d.pop("tool_calls")
        if self.tool_call_id is None:
            d.pop("tool_call_id")
        return d


@dataclass
class TaskState:
    """
    TaskState is the object provided from an Eval to a Solver. This must
    contain all the information that a Solver needs to provide a response to
    the Eval environment.

    Args:
        task_description: A string describing the task, including instructions
            and the expected response. Fixed across all instances of the eval.
        messages: The list of messages in the conversation so far. For example,
            it is often useful to include an input sample as the first message.
            Any previous interactions should also be included here.
        current_state: Any relevant state variables that should be passed to
            the Solver. While the current state of the eval should be apparent
            from previous messages, it is sometimes useful to include explicit
            state information here (e.g. the current game score or number of
            turns remaining) for easy access by the Solver without having to
            parse the messages.
    """

    task_description: str
    messages: list[Message] = field(default_factory=list)
    current_state: Any = None
