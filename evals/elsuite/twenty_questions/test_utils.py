from utils import format_msg, format_msgs
from evals.task_state import Message

def test_format_msg():
    msg = Message(content="I'm a message", role="guesser")

    assert format_msg(msg, "guesser") == Message(content="I'm a message", role="assistant")
    assert format_msg(msg, "gamemaster") == Message(content="I'm a message", role="user")

def test_format_msgs():
    msgs = [
        Message(content="I'm a guesser message", role="guesser"),
        Message(content="I'm a gamemaster message", role="gamemaster"),
        Message(content="I'm another guesser message", role="guesser"),
    ]

    assert format_msgs(msgs, "guesser") == [
        Message(content="I'm a guesser message", role="assistant"),
        Message(content="I'm a gamemaster message", role="user"),
        Message(content="I'm another guesser message", role="assistant"),
    ]
    
    assert format_msgs(msgs, "gamemaster") == [
        Message(content="I'm a guesser message", role="user"),
        Message(content="I'm a gamemaster message", role="assistant"),
        Message(content="I'm another guesser message", role="user"),
    ]