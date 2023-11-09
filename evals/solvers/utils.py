from evals.api import DummyCompletionFn
from evals.completion_fns.openai import OpenAIChatCompletionFn, OpenAICompletionFn
from evals.solvers.openai_chat_completion_solver import OpenAIChatCompletionSolver
from evals.solvers.openai_completion_solver import OpenAICompletionSolver
from evals.solvers.solver import DummySolver, Solver


def maybe_wrap_with_solver(completion_fn):
    """
    Converts a basic completion_fn into a Solver if it isn't already one.
    If it is already a Solver, it is returned unchanged.
    """

    if isinstance(completion_fn, Solver):
        # Use the solver directly
        solver = completion_fn
    else:
        # Wrap the completion_fn in an appropriate solver for its type
        if isinstance(completion_fn, OpenAIChatCompletionFn):
            solver = OpenAIChatCompletionSolver()
            solver.completion_fn = completion_fn
        elif isinstance(completion_fn, OpenAICompletionFn):
            solver = OpenAICompletionSolver()
            solver.completion_fn = completion_fn
        elif isinstance(completion_fn, DummyCompletionFn):
            solver = DummySolver()
        else:
            raise ValueError(f"Unsupported completion_fn type: {type(completion_fn)}")
    return solver
