from typing import Any, Union

from evals.api import CompletionFn, CompletionResult
from evals.prompt.base import OpenAICreateChatPrompt
from evals.solvers.nested.cot_solver import CoTSolver
from evals.solvers.solver import Solver, SolverSpec, create_solver
from evals.task_state import Message, TaskState


class SolverCompletionFnResult(CompletionResult):
    def __init__(self, msg):
        self.msg = msg

    def get_completions(self):
        return [self.msg]


class SolverCompletionFn(CompletionFn):
    """
    Wraps a solver into a completion function, s.t. that the completion function's
    __call__ method calls the internal solver's _solve method, mapping the input
    completion function `prompt` to the solver's `task_state` input.

    Useful for using Solvers with eval.Eval classes, which would normally require a CompletionFn.

    Current limitations:
        - Stateful solvers are not supported: Solver state is not maintained between
          calls.
        - Prompts with more than `role` and `content` keys are not supported.
    """

    def __init__(self, solver: Union[SolverSpec, Solver], registry: Any = None):
        if isinstance(solver, Solver):
            self.solver = solver
        else:
            self.solver = create_solver(solver)

    def __call__(
        self, prompt: Union[str, OpenAICreateChatPrompt], **kwargs
    ) -> SolverCompletionFnResult:
        # We have this check here rather than __init__ since the solver may be unwrapped and used in a SolverEval
        if isinstance(self.solver, CoTSolver):
            if self.solver.interaction_cache is not None:
                raise ValueError(
                    "`CoTSolver` with persistent memory is incompatible with "
                    "CompletionFn-based `Eval` classes. "
                    "Please set `CoTSolver(persistent_memory=False)` or update the eval to a `SolverEval`."
                )

        if isinstance(prompt, str):
            prompt = [{"role": "system", "content": prompt}]
        elif isinstance(prompt, list):
            assert prompt[0]["role"] == "system", "Unexpected prompt role ordering"
        else:
            raise ValueError(
                f"Unexpected prompt type: "
                f"string or OpenAICreateChatPrompt expected, got {type(prompt)}"
            )

        assert set(prompt[0].keys()) == {"role", "content",}, (
            "Unexpected keys in prompt: "
            f"expected exactly {{'role', 'content'}}, got {set(prompt[0].keys())}"
        )
        task_state = TaskState(
            prompt[0]["content"],
            [
                Message(msg["role"], msg["content"], msg.get("tool_calls"), msg.get("tool_call_id"))
                for msg in prompt[1:]
            ],
        )

        # use a copy to avoid task state surviving across samples
        pure_solver = self.solver.copy()

        result = pure_solver(task_state, **kwargs)
        return SolverCompletionFnResult(result.output)
