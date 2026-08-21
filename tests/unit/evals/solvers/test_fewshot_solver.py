from evals.solvers.nested.fewshot_solver import FewShotSolver
from evals.solvers.solver import Solver, SolverResult
from evals.task_state import TaskState


class RecordingSolver(Solver):
    def __init__(self) -> None:
        super().__init__()
        self.kwargs = None

    def _solve(self, task_state: TaskState, **kwargs) -> SolverResult:
        self.kwargs = kwargs
        return SolverResult("ok")


class PassthroughFewShotSolver(FewShotSolver):
    def __init__(self, base_solver: Solver) -> None:
        self._base_solver = base_solver

    @property
    def base_solver(self) -> Solver:
        return self._base_solver

    def _modify_task_state(self, task_state: TaskState) -> TaskState:
        return task_state


def test_fewshot_solver_forwards_runtime_kwargs() -> None:
    base_solver = RecordingSolver()
    solver = PassthroughFewShotSolver(base_solver)

    result = solver._solve(
        TaskState(task_description="test"),
        max_tokens=17,
        temperature=0.25,
        custom_option="value",
    )

    assert result.output == "ok"
    assert base_solver.kwargs == {
        "max_tokens": 17,
        "temperature": 0.25,
        "custom_option": "value",
    }
