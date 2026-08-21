from unittest.mock import patch

from evals.solvers.human_cli_solver import HumanCliSolver
from evals.solvers.solver import SolverResult
from evals.task_state import Message, TaskState


class UppercasePostprocessor:
    def __call__(self, result: SolverResult) -> SolverResult:
        return SolverResult(result.output.upper())


def _task_state() -> TaskState:
    return TaskState(
        task_description="Answer the task.",
        messages=[Message(role="user", content="What is 2 + 2?")],
    )


def test_default_mode_preserves_existing_single_prompt_input() -> None:
    solver = HumanCliSolver()
    expected_prompt = "system: Answer the task.\nuser: What is 2 + 2?\nassistant (you): "

    with (
        patch("builtins.input", return_value="4") as human_input,
        patch("evals.solvers.human_cli_solver.record_sampling") as record_sampling,
    ):
        result = solver(_task_state())

    assert result.output == "4"
    human_input.assert_called_once_with(expected_prompt)
    record_sampling.assert_called_once_with(
        prompt=expected_prompt,
        sampled="4",
        model="human",
    )


def test_explain_mode_shows_prompt_sampling_and_outputs(capsys) -> None:
    solver = HumanCliSolver(explain=True)
    solver.postprocessors = [UppercasePostprocessor()]
    expected_prompt = "system: Answer the task.\nuser: What is 2 + 2?\nassistant (you): "

    with (
        patch("builtins.input", return_value="four") as human_input,
        patch("evals.solvers.human_cli_solver.record_sampling") as record_sampling,
        patch("evals.solvers.solver.record_event"),
    ):
        result = solver(_task_state())

    output = capsys.readouterr().out
    assert result.output == "FOUR"
    assert "TASK CONTEXT (system)" in output
    assert "[0] user: What is 2 + 2?" in output
    assert "FINAL PROMPT STRING (exact)" in output
    assert expected_prompt in output
    assert "model: human" in output
    assert f"prompt_chars: {len(expected_prompt)}" in output
    assert "answer_chars: 4" in output
    assert "RAW OUTPUT" in output
    assert "four" in output
    assert "FINAL OUTPUT" in output
    assert "FOUR" in output
    human_input.assert_called_once_with("assistant (you): ")
    record_sampling.assert_called_once_with(
        prompt=expected_prompt,
        sampled="four",
        model="human",
    )
