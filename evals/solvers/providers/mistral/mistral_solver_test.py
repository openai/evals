import pytest

from evals.solvers.providers.mistral.mistral_solver import MistralSolver


@pytest.fixture
def solver():
    return MistralSolver(
        completion_fn_options={"model": "open-mistral-nemo"},
    )


def test_process_msgs_assistant_text_content_with_tool_calls(solver):
    messages = [
        {
            "role": "assistant",
            "content": "Here's the result",
            "tool_calls": [
                {"id": "call_123456789", "type": "function", "function": {"name": "test_func"}}
            ],
        }
    ]
    processed = solver._process_msgs(messages)
    assert len(processed) == 2
    assert processed[0] == {"role": "assistant", "content": "Here's the result"}
    assert processed[1] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call12345", "type": "function", "function": {"name": "test_func"}}],
    }


def test_process_msgs_assistant_empty_content_with_tool_calls(solver):
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_123456789", "type": "function", "function": {"name": "test_func"}}
            ],
        }
    ]
    processed = solver._process_msgs(messages)
    assert len(processed) == 1
    assert processed[0] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call12345", "type": "function", "function": {"name": "test_func"}}],
    }


def test_process_msgs_tool_message(solver):
    messages = [{"role": "tool", "content": "Tool result", "tool_call_id": "call_123456789"}]
    processed = solver._process_msgs(messages)
    assert len(processed) == 1
    assert processed[0] == {"role": "tool", "content": "Tool result", "tool_call_id": "call12345"}


def test_process_msgs_mixed_messages(solver):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "content": "Let me check",
            "tool_calls": [
                {"id": "call_987654321", "type": "function", "function": {"name": "check_func"}}
            ],
        },
        {"role": "tool", "content": "Check result", "tool_call_id": "call_987654321"},
        {"role": "assistant", "content": "Here's the answer"},
    ]
    processed = solver._process_msgs(messages)
    assert len(processed) == 6
    assert processed[0] == {"role": "system", "content": "You are a helpful assistant."}
    assert processed[1] == {"role": "user", "content": "Hello"}
    assert processed[2] == {"role": "assistant", "content": "Let me check"}
    assert processed[3] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call98765", "type": "function", "function": {"name": "check_func"}}],
    }
    assert processed[4] == {"role": "tool", "content": "Check result", "tool_call_id": "call98765"}
    assert processed[5] == {"role": "assistant", "content": "Here's the answer"}
