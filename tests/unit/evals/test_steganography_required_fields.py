import importlib
import sys
from types import ModuleType
from typing import Any


def _load_steganography(monkeypatch: Any):
    fake_metrics = ModuleType("evals.elsuite.steganography.reconstruction_metrics")
    fake_metrics.reconstruction_metrics = lambda _original, _recovered: {"semantic_similarity": 0.0}
    monkeypatch.setitem(
        sys.modules,
        "evals.elsuite.steganography.reconstruction_metrics",
        fake_metrics,
    )
    monkeypatch.delitem(sys.modules, "evals.elsuite.steganography.steganography", raising=False)
    return importlib.import_module("evals.elsuite.steganography.steganography")


def _run_with_responses(monkeypatch: Any, responses: list[str]):
    steganography = _load_steganography(monkeypatch)
    response_iter = iter(responses)
    monkeypatch.setattr(
        steganography,
        "run_completion_fn",
        lambda **_kwargs: (next(response_iter), "prompt"),
    )
    monkeypatch.setattr(
        steganography.monitor,
        "detect_suspicious_output",
        lambda *_args, **_kwargs: ("", 0),
    )

    return steganography.run_eval(
        task_prompt="task",
        reference_output="reference",
        payload="payload",
        completion_fn=object(),
        monitor_completion_fn=object(),
    )


def test_missing_encoder_task_output_is_a_rule_violation(monkeypatch: Any) -> None:
    result = _run_with_responses(
        monkeypatch,
        ['{"scratchpad": "notes"}', '{"payload": "payload"}'],
    )

    assert result["rule_violated"]
    assert result["model_output"] == "ERROR: encoder failed"


def test_missing_decoder_payload_is_a_rule_violation(monkeypatch: Any) -> None:
    result = _run_with_responses(
        monkeypatch,
        ['{"task_output": "answer"}', '{"scratchpad": "notes"}'],
    )

    assert result["rule_violated"]
    assert result["payload_out"] == "ERROR: decoder failed"
