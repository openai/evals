from typing import Any

from evals.solvers.prompts.cot import DEFAULT_COT_TEMPLATE, DEFAULT_EXTRACT_ANSWER_TEMPLATE
from evals.solvers.solver import NestedSolver, Solver, SolverResult, SolverSpec
from evals.solvers.utils import PersistentMemoryCache
from evals.task_state import Message, TaskState


class CoTSolver(NestedSolver):
    def __init__(
        self,
        cot_solver: SolverSpec,
        extract_solver: SolverSpec,
        cot_template: str = DEFAULT_COT_TEMPLATE,
        extract_template: str = DEFAULT_EXTRACT_ANSWER_TEMPLATE,
        persistent_memory: bool = True,
        private_interaction_length: int = 3,  # TODO: do this better
        registry: Any = None,
    ):
        super().__init__(cot_solver=cot_solver, extract_solver=extract_solver)

        self._cot_template = cot_template
        self._extract_template = extract_template

        self.interaction_cache = (
            PersistentMemoryCache(private_interaction_length) if persistent_memory else None
        )

    @property
    def cot_solver(self) -> Solver:
        return self.get_solver("cot_solver")

    @property
    def extract_solver(self) -> Solver:
        return self.get_solver("extract_solver")

    def cot_template(self, task_state: TaskState) -> str:
        #   This function is intended to be overwritten by solvers that extend CoTSolver
        #   and vary cot_template depending on the task_state
        return self._cot_template

    def extract_template(self, task_state: TaskState) -> str:
        #   This function is intended to be overwritten by solvers that extend CoTSolver
        #   and vary extract_template depending on the task_state
        return self._extract_template

    def _solve(
        self,
        task_state: TaskState,
        **kwargs,
    ) -> SolverResult:
        if self.interaction_cache:
            # Add in previous (private) memories
            task_state.messages = self.interaction_cache.load_private_interaction(task_state)

        # Reasoning step
        task_state.messages.append(Message(role="system", content=self.cot_template(task_state)))
        reasoning_result = self.cot_solver(task_state=task_state, **kwargs)
        reasoning_output = reasoning_result.output

        # Extract answer step
        task_state.messages.append(Message(role="assistant", content=reasoning_output))
        task_state.messages.append(
            Message(role="system", content=self.extract_template(task_state))
        )
        extracted_result = self.extract_solver(task_state=task_state, **kwargs)
        extracted_answer = extracted_result.output

        task_state.messages.append(Message(role="assistant", content=extracted_answer))

        # Save the interaction
        if self.interaction_cache:
            self.interaction_cache.save_private_interaction(task_state)

        return SolverResult(
            output=extracted_answer,
            reasoning_output=reasoning_output,
        )

    @property
    def name(self) -> str:
        return f"CoT_{self.cot_solver.name}_{self.extract_solver.name}"
