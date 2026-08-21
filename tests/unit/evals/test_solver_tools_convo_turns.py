from typing import Any


def _make_runner(output: str, max_turns: int, monkeypatch: Any):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    from evals.elsuite.solver_tools_convo import Runner
    from evals.solvers.solver import SolverResult

    class StaticSolver:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, task_state):
            self.calls += 1
            return SolverResult(output)

    solver = StaticSolver()
    runner = Runner(
        solver=solver,
        sample={"task": "Do the task", "answer": "done"},
        name_to_tool={},
        max_turns=max_turns,
        default_task_description="{tool_names_and_descriptions}",
        default_reminder_message="Try again",
    )
    return runner, solver


def test_exhausted_tool_run_reports_actual_turn_count(monkeypatch: Any) -> None:
    runner, solver = _make_runner("still thinking", max_turns=2, monkeypatch=monkeypatch)

    result = runner.run()

    assert solver.calls == 2
    assert result.metrics["num_turns"] == 2
    assert not result.metrics["is_correct"]


def test_final_answer_turn_count_is_unchanged(monkeypatch: Any) -> None:
    runner, solver = _make_runner("(@Answer: done)", max_turns=2, monkeypatch=monkeypatch)

    result = runner.run()

    assert solver.calls == 1
    assert result.metrics["num_turns"] == 1
    assert result.metrics["is_correct"]
