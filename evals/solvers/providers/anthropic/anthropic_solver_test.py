import os

import pytest
from anthropic.types import MessageParam, TextBlock, Usage

from evals.record import DummyRecorder
from evals.solvers.providers.anthropic.anthropic_solver import AnthropicSolver, anth_to_openai_usage
from evals.task_state import Message, TaskState

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
MODEL_NAME = "claude-instant-1.2"


@pytest.fixture
def anthropic_solver():
    solver = AnthropicSolver(
        model_name=MODEL_NAME,
    )
    return solver


@pytest.fixture
def dummy_recorder():
    """
    Sets the "default_recorder" necessary for sampling in the solver.
    """
    recorder = DummyRecorder(None)  # type: ignore
    with recorder.as_default_recorder("x"):
        yield recorder


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="API tests are wasteful to run on every commit.")
def test_solver(dummy_recorder, anthropic_solver):
    """
    Test that the solver generates a response coherent with the message history
    while following the instructions from the task description.
    - checks the task description is understood
    - checks that the messages are understood
    """
    solver = anthropic_solver

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
    Test that messages in our evals format are correctly
    converted to the format expected by Anthropic
    - "system" messages mapped to "user" in Anthropic
    - messages must alternate between "user" and "assistant"
    - messages are in MessageParam format
    """
    msgs = [
        Message(role="user", content="What is 2 + 2?"),
        Message(role="system", content="reason step by step"),
        Message(role="assistant", content="I don't need to reason for this, 2+2 is just 4"),
        Message(role="system", content="now, given your reasoning, provide the answer"),
    ]
    anth_msgs = AnthropicSolver._convert_msgs_to_anthropic_format(msgs)

    expected = [
        MessageParam(
            role="user",
            content=[
                TextBlock(text="What is 2 + 2?", type="text"),
                TextBlock(text="reason step by step", type="text"),
            ],
        ),
        MessageParam(
            role="assistant",
            content=[
                TextBlock(text="I don't need to reason for this, 2+2 is just 4", type="text"),
            ],
        ),
        MessageParam(
            role="user",
            content=[
                TextBlock(text="now, given your reasoning, provide the answer", type="text"),
            ],
        ),
    ]

    assert anth_msgs == expected, f"Expected {expected}, but got {anth_msgs}"


def test_anth_to_openai_usage_correctness():
    usage = Usage(input_tokens=100, output_tokens=150)
    expected = {
        "completion_tokens": 150,
        "prompt_tokens": 100,
        "total_tokens": 250,
    }
    assert (
        anth_to_openai_usage(usage) == expected
    ), "The conversion does not match the expected format."


def test_anth_to_openai_usage_zero_tokens():
    usage = Usage(input_tokens=0, output_tokens=0)
    expected = {
        "completion_tokens": 0,
        "prompt_tokens": 0,
        "total_tokens": 0,
    }
    assert anth_to_openai_usage(usage) == expected, "Zero token cases are not handled correctly."
