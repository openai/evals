from typing import Union

from evals.api import CompletionFn, DummyCompletionFn
from evals.completion_fns.openai import OpenAIChatCompletionFn, OpenAICompletionFn
from evals.completion_fns.solver_completion_fn import SolverCompletionFn
from evals.solvers.providers.openai.openai_solver import OpenAISolver
from evals.solvers.solver import DummySolver, Solver


def maybe_wrap_with_compl_fn(ambiguous_executor: Union[CompletionFn, Solver]) -> CompletionFn:
    """
    Converts a solver into a completion function if it isn't already one.
    If it is already a completion function, it is returned unchanged.
    """
    if isinstance(ambiguous_executor, Solver):
        completion_fn = SolverCompletionFn(solver=ambiguous_executor)
    elif isinstance(ambiguous_executor, CompletionFn):
        completion_fn = ambiguous_executor
    else:
        raise ValueError(
            f"Expected `executor` to be a `CompletionFn` or `Solver`, "
            f"but got {ambiguous_executor}"
        )

    return completion_fn


def maybe_wrap_with_solver(ambiguous_executor: Union[Solver, CompletionFn]) -> Solver:
    """
    Converts a basic completion_fn into a Solver if it isn't already one.
    If it is already a Solver, it is returned unchanged.
    """

    if isinstance(ambiguous_executor, Solver):
        # Use the solver directly
        solver = ambiguous_executor
    elif isinstance(ambiguous_executor, SolverCompletionFn):
        # unwrap previously wrapped solver
        solver = ambiguous_executor.solver
    else:
        # Wrap the completion_fn in an appropriate solver for its type
        if isinstance(ambiguous_executor, OpenAIChatCompletionFn) or isinstance(
            ambiguous_executor, OpenAICompletionFn
        ):
            solver = OpenAISolver(
                completion_fn_options={
                    "model": ambiguous_executor.model,
                }
            )
            solver.completion_fn = ambiguous_executor
        elif isinstance(ambiguous_executor, DummyCompletionFn):
            solver = DummySolver()
        else:
            raise ValueError(f"Unsupported completion_fn type: {type(ambiguous_executor)}")
    return solver
