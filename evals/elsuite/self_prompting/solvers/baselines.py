from evals.solvers.solver import Solver, SolverResult
from evals.task_state import TaskState


class BaselineNoPromptSolver(Solver):
    def __init__(
        self,
        **kwargs,
    ):
        """
        This solver simply returns an empty string as the prompt.
        """

    def _solve(
        self,
        task_state: TaskState,
        **kwargs,
    ) -> SolverResult:

        return SolverResult("")

    def name(self) -> str:
        return "SelfPromptingBaselineNoPromptSolver"


class BaselineOriginalPromptSolver(Solver):
    def __init__(
        self,
        **kwargs,
    ):
        """
        This solver simply returns the original instruction as the prompt.
        """

    def _solve(
        self,
        task_state: TaskState,
        **kwargs,
    ) -> SolverResult:

        instruction = task_state.current_state["instruction"]
        return SolverResult(instruction)

    def name(self) -> str:
        return "SelfPromptingBaselineOriginalPromptSolver"


class BaselineFewShotSolver(Solver):
    def __init__(
        self,
        **kwargs,
    ):
        """
        This solver concatenates the given input-output examples as few-shot demonstrations.
        """

    def _solve(
        self,
        task_state: TaskState,
        **kwargs,
    ) -> SolverResult:

        prompt = task_state.current_state["instruction"] + "\n"
        for sample in task_state.current_state["samples"]:
            prompt += f"""{sample["input"]}{sample["output"]}\n"""

        return SolverResult(prompt)

    def name(self) -> str:
        return "SelfPromptingBaselineFewShotSolver"
