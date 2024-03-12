import json
import math
import os
import tempfile
from pathlib import Path

import pytest

from evals.record import DummyRecorder
from evals.solvers.openai_assistants_solver import FILE_CACHE, OpenAIAssistantsSolver
from evals.task_state import Message, TaskState

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
MODEL = "gpt-4-1106-preview"

def test_print():
    print("IN_GITHUB_ACTIONS", IN_GITHUB_ACTIONS)
    raise NotImplementedError("This is a test error")

@pytest.fixture
def dummy_data_file(scope="session"):
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a data file
        dummy_data = {
            "passport": "12345678",
            "passepartout": "80",
            "password": "0netw0three",
        }
        tmpfile_path = str(Path(tmp_dir) / "password.json")
        json.dump(dummy_data, open(tmpfile_path, "w"))
        yield dummy_data, tmpfile_path


@pytest.fixture
def dummy_recorder():
    recorder = DummyRecorder(None)  # type: ignore
    with recorder.as_default_recorder("x"):
        yield recorder


@pytest.fixture
def vanilla_solver():
    solver = OpenAIAssistantsSolver(
        model=MODEL,
    )
    return solver


@pytest.fixture
def code_interpreter_solver():
    solver = OpenAIAssistantsSolver(
        model=MODEL,
        tools=[{"type": "code_interpreter"}],
    )
    return solver


@pytest.fixture
def retrieval_solver():
    solver = OpenAIAssistantsSolver(
        model=MODEL,
        tools=[{"type": "retrieval"}],
    )
    return solver


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="API tests are wasteful to run on every commit.")
def test_solver_copying(dummy_recorder, vanilla_solver):
    """
    When OpenAIAssistantsSolver is copied, the Assistant should be the same
    but the Thread should be different.
    """
    solver = vanilla_solver

    n_copies = 3
    for _ in range(n_copies):
        solver_copy = solver.copy()
        assert solver_copy.assistant.id == solver.assistant.id
        assert solver_copy.thread.id != solver.thread.id
        test_multiturn_conversation(dummy_recorder, solver_copy)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="API tests are wasteful to run on every commit.")
def test_multiturn_conversation(dummy_recorder, vanilla_solver):
    """
    Test that message history of the conversation is preserved across multiple turns.
    """
    solver = vanilla_solver

    numbers = [10, 13, 3, 6]
    input_messages = [Message(role="user", content=str(num)) for num in numbers]
    all_msgs = []
    for idx, msg in enumerate(input_messages):
        all_msgs.append(msg)
        solver_result = solver(
            TaskState(
                task_description="You will receive a sequence of numbers, please respond each time with the cumulative sum of all numbers sent so far. Answer with only a number.",
                messages=all_msgs,
            ),
        )
        print(solver_result.output)
        all_msgs.append(Message(role="assistant", content=solver_result.output))
        assert int(solver_result.output.strip()) == sum(numbers[: idx + 1])


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="API tests are wasteful to run on every commit.")
def test_code_interpreter(dummy_recorder, code_interpreter_solver):
    solver = code_interpreter_solver

    solver_result = solver(
        TaskState(
            task_description="",
            messages=[
                Message(
                    role="user", content="Please calculate the sqrt of 145145 to 3 decimal places."
                ),
            ],
        ),
    )
    print(solver_result.output)

    assert str(round(math.sqrt(145145), 3)) in solver_result.output


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="API tests are wasteful to run on every commit.")
def test_task_description(dummy_recorder, vanilla_solver):
    solver = vanilla_solver

    target_string = "Por favor, no hablo inglés."
    solver_result = solver(
        TaskState(
            task_description=f"Respond to all messages with '{target_string}'",  # Should overwrite the initial `instructions``
            messages=[
                Message(
                    role="user", content="Please calculate the sqrt of 145145 to 3 decimal places."
                ),
            ],
        ),
    )
    print(solver_result.output)
    assert solver_result.output == target_string


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="API tests are wasteful to run on every commit.")
def test_code_interpreter_file(dummy_recorder, dummy_data_file, code_interpreter_solver):
    dummy_data, tmpfile_path = dummy_data_file
    solver = code_interpreter_solver

    solver_result = solver(
        TaskState(
            task_description="",
            messages=[
                Message(
                    role="user",
                    content="Please return the value of the password in the attached file.",
                ),
            ],
            current_state={
                "files": [
                    tmpfile_path,
                ],
            },
        ),
    )
    print(solver_result.output)
    assert (
        dummy_data["password"] in solver_result.output
    ), f"Expected password '{dummy_data['password']}' to be in output, but got: {solver_result.output}"


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="API tests are wasteful to run on every commit.")
def test_retrieval_file(dummy_recorder, dummy_data_file, retrieval_solver):
    dummy_data, tmpfile_path = dummy_data_file
    solver = retrieval_solver

    solver_result = solver(
        TaskState(
            task_description="",
            messages=[
                Message(
                    role="user",
                    content="Please return the value of the password in the attached file.",
                ),
                # This prompt-hack is necessary for the model to actually use the file :(
                # We should be able to remove this in the future if the model improves.
                # https://community.openai.com/t/myfiles-browser-tool-is-not-operational-for-these-files/481922/18
                Message(
                    role="user",
                    content="Note that I have attached the file and it is accessible to you via the `myfiles_browser` tool.",
                ),
            ],
            current_state={
                "files": [
                    tmpfile_path,
                ],
            },
        ),
    )
    print(solver_result.output)
    assert (
        dummy_data["password"] in solver_result.output
    ), f"Expected password '{dummy_data['password']}' to be in output, but got: {solver_result.output}"


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="API tests are wasteful to run on every commit.")
def test_file_cache(dummy_recorder, dummy_data_file, retrieval_solver):
    dummy_data, tmpfile_path = dummy_data_file
    solver = retrieval_solver

    n_threads = 3
    solver_copies = [solver.copy() for _ in range(n_threads)]
    for solver_copy in solver_copies:
        test_retrieval_file(dummy_recorder, dummy_data_file, solver_copy)
        print()

    assert tmpfile_path in FILE_CACHE, f"File should be cached. Cached files: {FILE_CACHE}"
    cached_ids = [FILE_CACHE[tmpfile_path] for _ in solver_copies]
    assert all(
        [cached_id == FILE_CACHE[tmpfile_path] for cached_id in cached_ids]
    ), f"Cached file ID should be the same across threads, but got: {cached_ids}"
