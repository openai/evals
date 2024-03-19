from typing import Literal, List, Optional
from evals.task_state import Message, TaskState

PROMPTS = {
    "guesser": "You are about to play the game '20 questions'. The other player has thought of a noun which you should try to guess. You can ask 20 yes/no questions, to which they will answer 'yes', 'no', or 'skip' (if they cannot answer your question). You should try to guess the word as soon as possible, using the least amount of questions. To guess a word, type [GUESS <word>] – for example to guess the word 'sibling', output [GUESS sibling]. Your score will be 0 if you do not guess correctly, and {max_questions} minus the number of questions you asked if you guess correctly. Start by asking your first question.",
    "guesser_shortlist": "You are about to play the game '20 questions'. The other player has thought of a noun which you should try to guess. The noun is in this list: {shortlist}.\nYou can ask 20 yes/no questions, to which they will answer 'yes', 'no', or 'skip' (if they cannot answer your question). You should try to guess the word as soon as possible, using the least amount of questions. To guess a word, type [GUESS <word>] – for example to guess the word 'sibling', output [GUESS sibling]. Your score will be 0 if you do not guess correctly, and {max_questions} minus the number of questions you asked if you guess correctly. Start by asking your first question.",
    "gamemaster": "You are about to play the game '20 questions'. You'll be given a noun, which the other player will try to guess. They will ask 20 yes/no questions to try to guess the word. You will answer their questions with 'yes', 'no', or 'skip'. The word is: {word}.",
    "incorrect_guess": "Guess {guess} is incorrect. Ask more questions, or make another guess!",
    "rule_violation": "Your output was neither a guess nor a question. Try again! You can ask a yes/no question, or make a guess by outputting [GUESS <word>]."
}

def generate_task_state_for(role: Literal["guesser", "gamemaster"], conversation: list[Message], word: Optional[str] = None, max_questions: int = 20, shortlist: Optional[List[str]] = None) -> TaskState:
    """Generates a TaskState for the given role and conversation."""
    if role == "guesser":
        prompt = PROMPTS["guesser"].format(max_questions=max_questions) if shortlist is None else PROMPTS["guesser_shortlist"].format(max_questions=max_questions, shortlist=shortlist)
    elif role == "gamemaster":
        prompt = PROMPTS[role].format(word=word)
    else:
        raise ValueError(f"Invalid role: {role}")

    formatted_conversation = format_msgs(conversation, role)

    return TaskState(
        task_description=prompt,
        messages=formatted_conversation,
    )
    

def format_msgs(
    messages: list[Message],
    role: Literal["guesser", "gamemaster"],
) -> list[Message]:
    """Format messages from the perspective of the `role`."""
    new_messages = [format_msg(msg, role) for msg in messages]

    # post-conditions
    for m in new_messages:
        assert m.role in ["user", "assistant", "system"]

    return new_messages

def format_msg(msg: Message, role: Literal["guesser", "gamemaster"]) -> Message:
    """Formats a single message from the perspective of the `role`."""

    # body
    is_others_msg = role not in msg.role
    new_content = msg.content

    if is_others_msg:
        new_role = "user"
    elif is_system_msg(msg):
        new_role = "system"
    else:
        new_role = "assistant"

    new_message = Message(content=new_content, role=new_role)

    # post-conditions
    assert isinstance(new_message.content, str)
    assert new_message.role in ["user", "assistant", "system"]

    return new_message

def is_system_msg(m: Message) -> bool:
    assert isinstance(m, Message), "Message must be a Message type."
    assert hasattr(m, "role"), "Message must have a role."
    assert isinstance(m.role, str), "Message role must be a string."

    return m.role == "system"