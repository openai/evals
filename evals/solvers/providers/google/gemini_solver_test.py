import os

import pytest

from evals.record import DummyRecorder
from evals.solvers.providers.google.gemini_solver import GeminiSolver, GoogleMessage
from evals.task_state import Message, TaskState

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
MISSING_GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") in {None, ""}
MODEL_NAME = "gemini-pro"


@pytest.fixture
def dummy_recorder():
    recorder = DummyRecorder(None)  # type: ignore
    with recorder.as_default_recorder("x"):
        yield recorder


@pytest.fixture
def gemini_solver():
    os.environ["EVALS_SEQUENTIAL"] = "1"  # TODO: Remove after fixing threading issue
    solver = GeminiSolver(
        model_name=MODEL_NAME,
    )
    return solver


@pytest.mark.skipif(IN_GITHUB_ACTIONS or MISSING_GOOGLE_API_KEY, reason="API tests are wasteful to run on every commit.")
def test_solver(dummy_recorder, gemini_solver):
    """
    Test that the solver generates a response coherent with the message history
    while following the instructions from the task description.
    """
    solver = gemini_solver

    answer = "John Doe"
    task_state = TaskState(
        task_description=f"When you are asked for your name, respond with '{answer}' (without quotes).",
        messages=[
            Message(role="user", content="What is 2 + 2?"),
            Message(role="assistant", content="4"),
            Message(role="user", content="What is your name?"),
        ],
    )

    solver_res = solver(task_state=task_state)
    assert solver_res.output == answer, f"Expected '{answer}', but got {solver_res.output}"


def test_message_format():
    """
    Test that messages in our evals format is correctly converted to the format
    expected by Gemini.
    """

    messages = [
        Message(role="system", content="You are a great mathematician."),
        Message(role="user", content="What is 2 + 2?"),
        Message(role="assistant", content="5"),
        Message(role="user", content="That's incorrect. What is 2 + 2?"),
    ]

    gmessages = GeminiSolver._convert_msgs_to_google_format(messages)
    expected = [
        GoogleMessage(role="user", parts=["You are a great mathematician.\nWhat is 2 + 2?"]),
        GoogleMessage(role="model", parts=["5"]),
        GoogleMessage(role="user", parts=["That's incorrect. What is 2 + 2?"]),
    ]

    assert gmessages == expected, f"Expected {expected}, but got {gmessages}"
