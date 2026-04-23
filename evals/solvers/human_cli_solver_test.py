import builtins

import pytest

from evals.base import RunSpec
from evals.record import DummyRecorder
from evals.solvers.human_cli_solver import HumanCliSolver
from evals.task_state import Message, TaskState


@pytest.fixture
def dummy_recorder():
    yield DummyRecorder(
        RunSpec(
            completion_fns=[""],
            eval_name="",
            base_eval="",
            split="",
            run_config={},
            created_by="",
            run_id="",
            created_at="",
        )
    )


def test_human_cli_solver_default_mode_keeps_existing_behavior(dummy_recorder, monkeypatch, capsys):
    prompts = []

    def fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return "raw answer"

    monkeypatch.setattr(builtins, "input", fake_input)

    solver = HumanCliSolver()
    task_state = TaskState("Follow the instructions.", [Message("user", "Hello")])

    with dummy_recorder.as_default_recorder("x"):
        result = solver(task_state)

    assert result.output == "raw answer"
    assert prompts == ["system: Follow the instructions.\nuser: Hello\nassistant (you): "]
    assert capsys.readouterr().out == ""

    sampling_event = dummy_recorder.get_events("sampling")[0]
    assert sampling_event.data["prompt"] == prompts[0]
    assert sampling_event.data["sampled"] == "raw answer"
    assert sampling_event.data["model"] == "human"


def test_human_cli_solver_explain_mode_shows_context_and_postprocessed_output(
    dummy_recorder, monkeypatch, capsys
):
    prompts = []

    def fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return "  done.  "

    monkeypatch.setattr(builtins, "input", fake_input)

    solver = HumanCliSolver(
        explain=True,
        postprocessors=[
            "evals.solvers.postprocessors.postprocessors:Strip",
            "evals.solvers.postprocessors.postprocessors:RemovePeriod",
        ],
    )
    task_state = TaskState(
        "Answer as a human baseline.",
        [Message("user", "First question"), Message("assistant", "First answer")],
    )

    with dummy_recorder.as_default_recorder("x"):
        result = solver(task_state)

    assert result.output == "done"
    assert prompts == ["assistant (you): "]

    output = capsys.readouterr().out
    assert "================ TASK CONTEXT (system) ================" in output
    assert "Answer as a human baseline." in output
    assert "================ MESSAGE HISTORY ================" in output
    assert "[0] user: First question" in output
    assert "[1] assistant: First answer" in output
    assert "================ FINAL PROMPT STRING ================" in output
    assert "system: Answer as a human baseline." in output
    assert "assistant (you): " in output
    assert "================ SAMPLING RECORD ================" in output
    assert "model: human" in output
    assert "prompt_chars:" in output
    assert "answer_chars: 9" in output
    assert "================ OUTPUT ================" in output
    assert "raw_answer:   done.  " in output
    assert "final_answer: done" in output

    sampling_event = dummy_recorder.get_events("sampling")[0]
    assert sampling_event.data["sampled"] == "  done.  "
    postprocessor_events = dummy_recorder.get_events("postprocessor")
    assert len(postprocessor_events) == 2
