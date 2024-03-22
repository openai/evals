import pytest

from evals.solvers.together_solver import TogetherSolver


@pytest.fixture
def llama_solver():
    solver = TogetherSolver(
        completion_fn_options={
            "model": "meta-llama/Llama-2-13b-chat-hf",
        },
    )
    return solver


@pytest.fixture
def llama_solver_merge():
    solver = TogetherSolver(
        merge_adjacent_msgs=True,
        completion_fn_options={
            "model": "meta-llama/Llama-2-13b-chat-hf",
        },
    )
    return solver


def test_single_system_msg(llama_solver):
    in_msgs = [
        {"role": "system", "content": "Hello"},
    ]
    out_msgs = [
        {"role": "user", "content": "Hello"},
    ]
    assert llama_solver._process_msgs(in_msgs) == out_msgs


def test_system_assistant_msgs(llama_solver):
    in_msgs = [
        {"role": "system", "content": "Hello"},
        {"role": "assistant", "content": "Hi, how are ya?"},
    ]
    out_msgs = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi, how are ya?"},
    ]
    assert llama_solver._process_msgs(in_msgs) == out_msgs


def test_system_user_msg(llama_solver):
    in_msgs = [
        {"role": "system", "content": "Hello"},
        {"role": "user", "content": "Hi, how are ya?"},
    ]
    out_msgs = [
        {"role": "system", "content": "Hello"},
        {"role": "user", "content": "Hi, how are ya?"},
    ]
    assert llama_solver._process_msgs(in_msgs) == out_msgs


def test_final_system_msg(llama_solver):
    in_msgs = [
        {"role": "system", "content": "Hello"},
        {"role": "user", "content": "Hi, how are ya?"},
        {"role": "system", "content": "Good, you?"},
    ]
    out_msgs = [
        {"role": "system", "content": "Hello"},
        {"role": "user", "content": "Hi, how are ya?"},
        {"role": "user", "content": "Good, you?"},
    ]
    assert llama_solver._process_msgs(in_msgs) == out_msgs


def test_combined(llama_solver):
    in_msgs = [
        {"role": "system", "content": "Hello"},
        {"role": "assistant", "content": "Hi, how are ya?"},
        {"role": "system", "content": "Good, you?"},
    ]
    out_msgs = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi, how are ya?"},
        {"role": "user", "content": "Good, you?"},
    ]
    assert llama_solver._process_msgs(in_msgs) == out_msgs


def test_merge(llama_solver_merge):
    in_msgs = [
        {"role": "system", "content": "Hello"},
        {"role": "user", "content": "Hi, how are ya?"},
        {"role": "user", "content": "Good, you?"},
    ]
    out_msgs = [
        {"role": "system", "content": "Hello"},
        {"role": "user", "content": "Hi, how are ya?\n\nGood, you?"},
    ]
    assert llama_solver_merge._process_msgs(in_msgs) == out_msgs


def test_advanced_merge(llama_solver_merge):
    in_msgs = [
        {"role": "system", "content": "Hello"},
        {"role": "user", "content": "Hi, how are ya?"},
        {"role": "user", "content": "Good, you?"},
        {"role": "assistant", "content": "Message 1"},
        {"role": "assistant", "content": "Message 2"},
        {"role": "user", "content": "Message 3"},
    ]
    out_msgs = [
        {"role": "system", "content": "Hello"},
        {"role": "user", "content": "Hi, how are ya?\n\nGood, you?"},
        {"role": "assistant", "content": "Message 1\n\nMessage 2"},
        {"role": "user", "content": "Message 3"},
    ]
    assert llama_solver_merge._process_msgs(in_msgs) == out_msgs
