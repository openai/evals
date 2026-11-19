import json
import sys
import types

import pytest

# Avoid loading the heavyweight sentence-embedding model during tests.
module_name = "evals.elsuite.steganography.reconstruction_metrics"
previous_module = sys.modules.get(module_name)

mock_rm = types.ModuleType(module_name)
mock_rm.reconstruction_metrics = lambda original, decompressed: {
    "semantic_similarity": 1.0
}
sys.modules[module_name] = mock_rm

from evals.elsuite.steganography import steganography

# Restore the original module state after importing steganography.
if previous_module is None:
    sys.modules.pop(module_name, None)
else:
    sys.modules[module_name] = previous_module


class FakeCompletionResult:
    def __init__(self, response):
        self.response = response

    def get_completions(self):
        return [self.response]


class FakeCompletionFn:
    def __init__(self, *responses):
        self.responses = list(responses)

    def __call__(self, prompt):
        return FakeCompletionResult(self.responses.pop(0))


@pytest.fixture(autouse=True)
def mock_monitor(monkeypatch):
    monkeypatch.setattr(
        steganography.monitor,
        "detect_suspicious_output",
        lambda *args: ("", False),
    )


def run_test_eval(completion_fn):
    return steganography.run_eval(
        task_prompt="Write a short sentence.",
        reference_output="A reference sentence.",
        payload="hidden message",
        completion_fn=completion_fn,
        monitor_completion_fn=completion_fn,
        prompt_version="taskonly",
    )


def test_valid_required_fields_are_accepted():
    completion_fn = FakeCompletionFn(
        json.dumps({"task_output": "Encoded output"}),
        json.dumps({"payload": "hidden message"}),
    )

    result = run_test_eval(completion_fn)

    assert result["model_output"] == "Encoded output"
    assert result["payload_out"] == "hidden message"
    assert result["rule_violated"] is False


def test_missing_task_output_is_rejected():
    completion_fn = FakeCompletionFn(
        json.dumps({"scratchpad": "Encoding explanation"}),
        json.dumps({"payload": "hidden message"}),
    )

    result = run_test_eval(completion_fn)

    assert result["rule_violated"] is True
    assert "Encoder failed" in result["rule_violations"]


def test_missing_payload_is_rejected():
    completion_fn = FakeCompletionFn(
        json.dumps({"task_output": "Encoded output"}),
        json.dumps({"scratchpad": "Decoding explanation"}),
    )

    result = run_test_eval(completion_fn)

    assert result["rule_violated"] is True
    assert "Decoder failed" in result["rule_violations"]


def test_scratchpad_fields_are_optional():
    completion_fn = FakeCompletionFn(
        json.dumps({"task_output": "Encoded output"}),
        json.dumps({"payload": "hidden message"}),
    )

    result = run_test_eval(completion_fn)

    assert result["encode_scratchpad"] == ""
    assert result["decode_scratchpad"] == ""
    assert result["rule_violated"] is False
