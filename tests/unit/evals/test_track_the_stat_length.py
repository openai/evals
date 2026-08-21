from typing import Any


def _make_eval(monkeypatch: Any):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    from evals.elsuite.track_the_stat import utils
    from evals.elsuite.track_the_stat.eval import TrackTheStat

    evaluator = object.__new__(TrackTheStat)
    evaluator.task = "median"
    evaluator.task_desc = ""
    evaluator.task_fn = utils.median
    return evaluator


def test_track_the_stat_counts_fully_successful_sequence(monkeypatch: Any) -> None:
    from evals.solvers.solver import SolverResult

    class PerfectSolver:
        def __call__(self, task_state):
            values = task_state.current_state["state_data"]
            values = sorted(values)
            middle = len(values) // 2
            if len(values) % 2:
                answer = values[middle]
            else:
                answer = (values[middle - 1] + values[middle]) / 2
            return SolverResult(f"[median: {answer}]")

    evaluator = _make_eval(monkeypatch)
    result = evaluator._eval_sample(PerfectSolver(), [1, 5, 3])

    assert result == {"max_length": 3, "violation": False}


def test_track_the_stat_keeps_early_failure_count(monkeypatch: Any) -> None:
    from evals.solvers.solver import SolverResult

    class FailSecondSolver:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, task_state):
            self.calls += 1
            if self.calls == 1:
                return SolverResult("[median: 1]")
            return SolverResult("[median: 999]")

    evaluator = _make_eval(monkeypatch)
    result = evaluator._eval_sample(FailSecondSolver(), [1, 5, 3])

    assert result == {"max_length": 1, "violation": False}
