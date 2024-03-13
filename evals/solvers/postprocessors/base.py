from abc import ABC, abstractmethod

from evals.solvers.solver import SolverResult


class PostProcessor(ABC):
    """
    A postprocessor is a class that processes the output of a solver.
    It is used to extract the relevant information from the output of the solver.
    """

    @abstractmethod
    def __call__(self, result: SolverResult, *args, **kwargs) -> SolverResult:
        """
        Process the result of the solver.
        """
        raise NotImplementedError
