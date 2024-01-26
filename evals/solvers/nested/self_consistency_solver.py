import logging
from collections import Counter
from typing import Any, Optional

from evals.solvers.solver import NestedSolver, Solver, SolverResult, SolverSpec
from evals.solvers.utils import PersistentMemoryCache
from evals.task_state import Message, TaskState

DEFAULT_COT_TEMPLATE = """Before answering, reason in a step-by-step manner as to get the right answer, then conclude with the answer. Format your output as {prefix} <answer>"""
DEFAULT_ANSWER_PREFIX = "The answer is"
DEFAULT_JUDGE_PROMPT = "You are about to see some answers for the follwing question: '{question}'. If there is agreement between the answers, output the consensus answer. Otherwise, output [NO CONSENSUS]. You should format your final answer as {prefix} <answer>"

logger = logging.getLogger(__name__)


class SelfConsistencySolver(NestedSolver):
    """This is a solver that implements self-consistency prompting.
    It works by generating multiple chain-of-thought completions, and
    selecting the answer that occurs most frequently in the completions.
    The answer in each completion is extracted by looking for a prefix,
    either the default above or one provided through the YAML config.
    """

    def __init__(
        self,
        solver: SolverSpec,
        num_generations: int = 5,
        cot_template: str = DEFAULT_COT_TEMPLATE,
        answer_prefix: str = DEFAULT_ANSWER_PREFIX,
        judge_prompt: Optional[str] = None,
        mode: str = "count",
        persistent_memory: bool = True,
        private_interaction_length: int = 1,
        registry: Any = None,
    ):
        super().__init__(registry=registry, solver=solver, judge_solver=solver)
        self.num_generations = num_generations
        self.answer_prefix = answer_prefix
        self.cot_template = cot_template.format(prefix=self.answer_prefix)
        self.mode = mode
        self.judge_prompt = judge_prompt if judge_prompt else DEFAULT_JUDGE_PROMPT

        # Every time a private interaction happens, we cache one
        # additional prompt and num_generations reasoning completions.
        self.interaction_cache = (
            PersistentMemoryCache(private_interaction_length + num_generations)
            if persistent_memory
            else None
        )

    @property
    def solver(self) -> Solver:
        return self.get_solver("solver")

    @property
    def judge_solver(self) -> Solver:
        return self.get_solver("judge_solver")

    def _solve(
        self,
        task_state: TaskState,
        **kwargs,
    ) -> SolverResult:
        unique_answers = Counter()

        if self.interaction_cache:
            # Add in previous (private) memories
            task_state.messages = self.interaction_cache.load_private_interaction(task_state)

        # Execute reasoning step by generating multiple reasoning paths.
        task_state.messages.append(Message(role="system", content=self.cot_template))
        reasoning_completions = []

        for i in range(self.num_generations):
            raw_result = self.solver(task_state)

            # We don't immediately append this to the task state messages because
            # doing so would influence subsequent reasonings, which we do not want.
            reasoning_completions.append(raw_result.output)
            try:
                answer = self._extract_answer(raw_result)
                unique_answers[answer] += 1
            except ValueError as ve:
                logger.info(f"ValueError while extracting answer: {ve}")
                continue

        # Extract the consensus answer from all the reasonings, if possible to do so.
        if self.mode == "count":
            if len(unique_answers) > 0:
                consensus_answer, num_occurrences = unique_answers.most_common(1)[0]
            else:
                logger.error(
                    f"Could not detect any answers for mode 'count' among the completions: {reasoning_completions}"
                )
        else:
            if len(task_state.messages) > 0:
                prompt = task_state.messages[-2].content
            else:
                prompt = task_state.task_description
            previous_answer_task_state = TaskState(
                task_description=task_state.task_description,
                messages=[
                    *task_state.messages,  # Previous message history
                    Message(
                        role="system",
                        content=self.judge_prompt.format(
                            question=prompt, prefix=self.answer_prefix
                        ),
                    ),
                    *[Message(role="user", content=answer) for answer in reasoning_completions],
                    Message(
                        role="system",
                        content="What is the final answer in the format requested by the question?",
                    ),
                ],
            )
            raw_result = self.judge_solver(previous_answer_task_state)
            try:
                consensus_answer = self._extract_answer(raw_result)
            except ValueError as ve:
                logger.info(f"ValueError while extracting answer: {ve}")
                consensus_answer = "[NO CONSENSUS]"

        task_state.messages += reasoning_completions
        if self.interaction_cache:
            self.interaction_cache.save_private_interaction(task_state)

        return SolverResult(
            output=consensus_answer,
            reasoning_completions=reasoning_completions,
        )

    def _extract_answer(self, raw_result: SolverResult) -> str:
        out_text = raw_result.output
        if self.answer_prefix.lower() in out_text:
            answer = out_text.split(self.answer_prefix.lower())[1].strip()
        elif self.answer_prefix in out_text:
            answer = out_text.split(self.answer_prefix)[1].strip()
        else:
            raise ValueError(
                f"Failed to extract a valid answer: answer prefix '{self.answer_prefix}' not found in model response: {out_text}."
            )
        return answer

    @property
    def name(self) -> str:
        return f"SelfConsistencySolver wrapping {self.solver.name}"
