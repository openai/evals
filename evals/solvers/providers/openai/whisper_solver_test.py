import pytest

from evals.solvers.providers.openai.whisper_solver import WhisperCascadedSolver
from evals.task_state import Message


@pytest.fixture
def whisper_solver():
    solver_spec = {"class": "evals.solvers.solver:DummySolver", "args": {}}
    solver = WhisperCascadedSolver(solver=solver_spec)
    solver._transcribe = lambda x: "Hello World!"
    return solver


TEST_TRANSCRIPT = "Hello World!"
TEST_PART_AUDIO = {
    "type": "image_url",
    "image_url": {"url": "data:audio/x-wav;base64,SGVsbG8gV29ybGQh"},
}
TEST_PART_PROLOG = {"type": "text", "text": "Prolog..."}
TEST_PART_EPILOG = {"type": "text", "text": "Epilog..."}


def test_string_content(whisper_solver):
    in_msgs = [Message(role="user", content="Test Message")]
    out_msgs = [Message(role="user", content="Test Message")]
    assert whisper_solver._process_msgs(in_msgs) == out_msgs


def test_audio_only(whisper_solver):
    in_msgs = [Message(role="user", content=[TEST_PART_AUDIO])]
    out_msgs = [Message(role="user", content=TEST_TRANSCRIPT)]
    assert whisper_solver._process_msgs(in_msgs) == out_msgs


def test_text_then_audio(whisper_solver):
    in_msgs = [Message(role="user", content=[TEST_PART_PROLOG, TEST_PART_AUDIO])]
    out_msgs = [Message(role="user", content="Prolog...\nHello World!")]
    assert whisper_solver._process_msgs(in_msgs) == out_msgs


def test_audio_then_text(whisper_solver):
    in_msgs = [Message(role="user", content=[TEST_PART_AUDIO, TEST_PART_EPILOG])]
    out_msgs = [Message(role="user", content="Hello World!\nEpilog...")]
    assert whisper_solver._process_msgs(in_msgs) == out_msgs


def test_text_then_audio_then_text(whisper_solver):
    in_msgs = [Message(role="user", content=[TEST_PART_PROLOG, TEST_PART_AUDIO, TEST_PART_EPILOG])]
    out_msgs = [Message(role="user", content="Prolog...\nHello World!\nEpilog...")]
    assert whisper_solver._process_msgs(in_msgs) == out_msgs
