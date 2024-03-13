import pytest

from evals.record import DummyRecorder
from evals.solvers.solver import Solver, SolverResult
from evals.task_state import TaskState


class EchoSolver(Solver):
    """
    A solver that simply returns the task description.
    """

    def _solve(
        self,
        task_state: TaskState,
        **kwargs,
    ) -> SolverResult:
        return SolverResult(task_state.task_description)


@pytest.fixture
def dummy_recorder():
    recorder = DummyRecorder(None)  # type: ignore
    with recorder.as_default_recorder("x"):
        yield recorder


def test_echo_solver(dummy_recorder):
    text = "Please directly echo this text."
    task_state = TaskState(text, [])
    solver = EchoSolver()
    result = solver(task_state)
    assert result.output == text


def test_echo_solver_with_postprocessors(dummy_recorder):
    text = "p@ssw0rd!"

    task_state = TaskState(f"   {text}\n\n  ", [])
    solver = EchoSolver(postprocessors=["evals.solvers.postprocessors.postprocessors:Strip"])
    result = solver(task_state)
    assert result.output == text

    task_state = TaskState(f"'{text}'", [])
    solver = EchoSolver(postprocessors=["evals.solvers.postprocessors.postprocessors:RemoveQuotes"])
    result = solver(task_state)
    assert result.output == text

    task_state = TaskState(f"{text}.", [])
    solver = EchoSolver(postprocessors=["evals.solvers.postprocessors.postprocessors:RemovePeriod"])
    result = solver(task_state)
    assert result.output == text

    task_state = TaskState(f"   '{text}'  ", [])
    solver = EchoSolver(
        postprocessors=[
            "evals.solvers.postprocessors.postprocessors:Strip",
            "evals.solvers.postprocessors.postprocessors:RemoveQuotes",
        ]
    )
    result = solver(task_state)
    assert result.output == text

    task_state = TaskState(f"\n'{text}.'\n", [])
    solver = EchoSolver(
        postprocessors=[
            "evals.solvers.postprocessors.postprocessors:Strip",
            "evals.solvers.postprocessors.postprocessors:RemoveQuotes",
            "evals.solvers.postprocessors.postprocessors:RemovePeriod",
        ]
    )
    result = solver(task_state)
    assert result.output == text
