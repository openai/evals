from evals.solvers.nested.self_consistency_solver import SelfConsistencySolver
from evals.solvers.solver import Solver, SolverResult
from evals.task_state import Message, TaskState


class StaticSolver(Solver):
    def __init__(self, output: str) -> None:
        super().__init__()
        self.output = output
        self.calls = 0

    def _solve(self, task_state: TaskState, **kwargs) -> SolverResult:
        self.calls += 1
        return SolverResult(self.output)


class StubSelfConsistencySolver(SelfConsistencySolver):
    def __init__(self, child_solver: Solver) -> None:
        self._child_solver = child_solver
        self.num_generations = 3
        self.answer_prefix = "The answer is"
        self.cot_template = "Reason, then provide the final answer."
        self.mode = "count"
        self.judge_prompt = "unused"
        self.interaction_cache = None

    @property
    def solver(self) -> Solver:
        return self._child_solver

    @property
    def judge_solver(self) -> Solver:
        return self._child_solver


def test_count_mode_returns_no_consensus_when_no_answer_can_be_extracted() -> None:
    child_solver = StaticSolver("I could not determine a final answer.")
    solver = StubSelfConsistencySolver(child_solver)

    result = solver(
        TaskState(
            task_description="Answer the question.",
            messages=[Message(role="user", content="What is the answer?")],
        )
    )

    assert result.output == "[NO CONSENSUS]"
    assert result.metadata["reasoning_completions"] == [
        "I could not determine a final answer.",
        "I could not determine a final answer.",
        "I could not determine a final answer.",
    ]
    assert child_solver.calls == 3
