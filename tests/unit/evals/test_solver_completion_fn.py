from typing import Optional

from evals.completion_fns.solver_completion_fn import SolverCompletionFn
from evals.solvers.solver import Solver, SolverResult
from evals.task_state import Message, TaskState


class RecordingSolver(Solver):
    def __init__(self) -> None:
        super().__init__()
        self.task_state: Optional[TaskState] = None

    def _solve(self, task_state: TaskState, **kwargs) -> SolverResult:
        self.task_state = task_state
        return SolverResult("ok")

    def copy(self) -> "RecordingSolver":
        return self


def test_solver_completion_fn_accepts_user_first_chat_prompt() -> None:
    solver = RecordingSolver()
    completion_fn = SolverCompletionFn(solver)

    result = completion_fn(
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
    )

    assert result.get_completions() == ["ok"]
    assert solver.task_state == TaskState(
        task_description="",
        messages=[
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi"),
        ],
    )


def test_solver_completion_fn_preserves_system_first_behavior() -> None:
    solver = RecordingSolver()
    completion_fn = SolverCompletionFn(solver)

    completion_fn(
        [
            {"role": "system", "content": "Follow the rules"},
            {"role": "user", "content": "Question"},
        ]
    )

    assert solver.task_state == TaskState(
        task_description="Follow the rules",
        messages=[Message(role="user", content="Question")],
    )
