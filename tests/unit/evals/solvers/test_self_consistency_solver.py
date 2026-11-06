from copy import deepcopy

import pytest

from evals.solvers.nested.self_consistency_solver import SelfConsistencySolver
from evals.solvers.solver import Solver, SolverResult
from evals.task_state import Message, TaskState


class ScriptedSolver(Solver):
    def __init__(self, outputs: list[str]) -> None:
        super().__init__()
        self.outputs = outputs
        self.calls = []

    def _solve(self, task_state: TaskState, **kwargs) -> SolverResult:
        output = self.outputs[len(self.calls)]
        self.calls.append(deepcopy(task_state))
        return SolverResult(output)


def make_solver(outputs, **kwargs):
    return SelfConsistencySolver(
        solver={"class": f"{__name__}:ScriptedSolver", "args": {"outputs": outputs}},
        num_generations=len(outputs),
        **kwargs,
    )


def make_task():
    return TaskState(
        task_description="Answer the question.",
        messages=[Message(role="user", content="What is the answer?")],
    )


@pytest.mark.parametrize("persistent_memory", [False, True])
@pytest.mark.parametrize("outputs", [["No answer."], ["Unclear.", "Cannot determine.", ""]])
def test_count_mode_returns_no_consensus_with_no_extractable_answer(
    outputs, persistent_memory, caplog
):
    solver = make_solver(outputs, persistent_memory=persistent_memory)
    task = make_task()
    original = deepcopy(task)

    result = solver(task)

    assert result.output == "[NO CONSENSUS]"
    assert result.metadata == {"reasoning_completions": outputs}
    assert len(solver.solver.calls) == len(outputs)
    assert solver.judge_solver.calls == []
    assert task == original
    assert "Could not detect any answers" in caplog.text
    assert repr(outputs) in caplog.text
    if persistent_memory:
        assert solver.interaction_cache.last_interaction is not None
    else:
        assert solver.interaction_cache is None


@pytest.mark.parametrize(
    "outputs, expected",
    [
        (["The answer is A", "The answer is B", "The answer is A"], "A"),
        (["Invalid", "The answer is B", "Invalid again"], "B"),
        (["The answer is B", "The answer is A"], "B"),
        (["the answer is C", "The answer is C"], "C"),
    ],
)
def test_count_mode_preserves_voting_and_partial_failure_behavior(outputs, expected):
    solver = make_solver(outputs, persistent_memory=False)
    result = solver(make_task())
    assert result.output == expected
    assert result.metadata["reasoning_completions"] == outputs
    assert len(solver.solver.calls) == len(outputs)
    assert solver.judge_solver.calls == []
    # Independent generations see the same prompt rather than earlier answers.
    assert all(task == solver.solver.calls[0] for task in solver.solver.calls)


def test_custom_answer_prefix_is_preserved():
    solver = make_solver(["Final: Z", "Invalid"], answer_prefix="Final:", persistent_memory=False)
    assert solver(make_task()).output == "Z"


@pytest.mark.parametrize(
    "judge_output, expected", [("Undecided.", "[NO CONSENSUS]"), ("The answer is Z", "Z")]
)
def test_judge_mode_retains_its_existing_results(judge_output, expected):
    outputs = ["The answer is A", "The answer is B"]
    solver = make_solver(outputs, mode="judge", persistent_memory=False)
    solver._solver_cache["judge_solver"] = ScriptedSolver([judge_output])
    result = solver(make_task())
    assert result.output == expected
    assert result.metadata["reasoning_completions"] == outputs
    assert len(solver.judge_solver.calls) == 1
