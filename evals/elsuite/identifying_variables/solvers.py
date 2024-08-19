import random

from evals.solvers.solver import Solver, SolverResult
from evals.task_state import TaskState


class RandomSolver(Solver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _solve(self, task_state: TaskState) -> SolverResult:
        valid_hyp = random.uniform(0, 1) < 0.5

        variables = task_state.current_state["variables"]
        n_vars_to_sample = random.randint(2, len(variables))
        ind_var, dep_var, *ctrl_vars = random.sample(variables, n_vars_to_sample)
        if len(ctrl_vars) == 0:
            ctrl_vars = "none"
        else:
            ctrl_vars = ", ".join(ctrl_vars)

        solver_string = f"[@ANSWER valid_hyp: {valid_hyp}; independent: {ind_var}; dependent: {dep_var}; control: {ctrl_vars}]"

        return SolverResult(output=solver_string)


class NoCtrl(Solver):
    """
    Solver that always returns no control variables
    (i.e. "none", interpreted as an empty list by the eval)
    what it returns for the other variables is arbitrary
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _solve(self, task_state: TaskState) -> SolverResult:
        # we don't care about valid_hyp and ind/dep vars for this solver
        # it's only used for the ctrl variables subtask
        valid_hyp = True
        variables = task_state.current_state["variables"]
        ind_var, dep_var = random.sample(variables, 2)

        # it just always returns no control variables
        ctrl_vars = "none"
        solver_string = f"[@ANSWER valid_hyp: {valid_hyp}; independent: {ind_var}; dependent: {dep_var}; control: {ctrl_vars}]"

        return SolverResult(output=solver_string)
