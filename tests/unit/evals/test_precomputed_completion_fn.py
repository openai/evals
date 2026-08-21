import pytest

from evals.base import RunSpec
from evals.completion_fns.precomputed import PrecomputedCompletionFn
from evals.record import RecorderBase


def make_recorder() -> RecorderBase:
    return RecorderBase(
        RunSpec(
            completion_fns=["precomputed"],
            eval_name="test.dev",
            base_eval="test",
            split="dev",
            run_config={},
            created_by="test",
        )
    )


def test_precomputed_completion_supports_string_and_chat_prompts() -> None:
    samples = [
        {"input": "what is 2+1?", "output": "3", "ideal": "3"},
        {
            "input": [{"role": "user", "content": "capital of France?"}],
            "output": "Paris",
            "ideal": "Paris",
        },
    ]
    completion_fn = PrecomputedCompletionFn(samples)
    recorder = make_recorder()

    with recorder.as_default_recorder("test.dev.0"):
        assert completion_fn("what is 2+1?").get_completions() == ["3"]

    with recorder.as_default_recorder("test.dev.1"):
        assert completion_fn(samples[1]["input"]).get_completions() == ["Paris"]

    sampling_events = recorder.get_events("sampling")
    assert [event.data["sampled"] for event in sampling_events] == [["3"], ["Paris"]]
    assert all(event.data["model"] == "precomputed" for event in sampling_events)


def test_precomputed_completion_supports_custom_output_key() -> None:
    completion_fn = PrecomputedCompletionFn(
        [{"input": "prompt", "prediction": "stored response"}],
        output_key="prediction",
    )
    recorder = make_recorder()

    with recorder.as_default_recorder("test.dev.0"):
        assert completion_fn("prompt").get_completions() == ["stored response"]


def test_precomputed_completion_rejects_missing_output() -> None:
    with pytest.raises(ValueError, match="missing 'output'"):
        PrecomputedCompletionFn([{"input": "prompt", "ideal": "answer"}])


def test_precomputed_completion_rejects_ambiguous_duplicate_inputs() -> None:
    with pytest.raises(ValueError, match="same input with different outputs"):
        PrecomputedCompletionFn(
            [
                {"input": "same prompt", "output": "first"},
                {"input": "same prompt", "output": "second"},
            ]
        )


def test_precomputed_completion_errors_when_eval_changes_prompt() -> None:
    completion_fn = PrecomputedCompletionFn([{"input": "original", "output": "answer"}])
    recorder = make_recorder()

    with recorder.as_default_recorder("test.dev.0"), pytest.raises(
        KeyError, match="No precomputed output matches this prompt"
    ):
        completion_fn("modified prompt")
