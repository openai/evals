from pathlib import Path

from evals.elsuite.basic.match_with_solvers import MatchWithSolvers
from evals.record import DummyRecorder
from evals.solvers.solver import DummySolver, Solver, SolverResult
from evals.task_state import TaskState


class StaticSolver(Solver):
    def __init__(self, output: str) -> None:
        super().__init__()
        self.output = output

    def _solve(self, task_state: TaskState, **kwargs) -> SolverResult:
        return SolverResult(self.output)


def _eval() -> MatchWithSolvers:
    return MatchWithSolvers(
        completion_fns=[DummySolver()],
        samples_jsonl="",
        task_description="Answer the question.",
        eval_registry_path=Path("."),
    )


def test_match_with_solvers_accepts_non_first_ideal() -> None:
    evaluator = _eval()
    recorder = DummyRecorder(None)
    sample = {
        "input": [{"role": "user", "content": "Question"}],
        "ideal": ["first", "second"],
    }

    with recorder.as_default_recorder("x"):
        picked = evaluator.eval_sample(StaticSolver("second"), sample, None)

    assert picked == "second"


def test_match_with_solvers_capitalizes_every_ideal() -> None:
    evaluator = _eval()
    recorder = DummyRecorder(None)
    sample = {
        "input": [{"role": "user", "content": "Question"}],
        "ideal": ["first", "second"],
    }

    with recorder.as_default_recorder("x"):
        picked = evaluator.eval_sample(StaticSolver("Second"), sample, None)

    assert picked == "Second"
