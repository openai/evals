import json
from abc import ABC, abstractmethod
from copy import deepcopy
from importlib import import_module
from typing import Any, Dict, TypeVar, Union

from pydantic import TypeAdapter, ValidationError
from typing_extensions import TypedDict

from evals.api import CompletionFn
from evals.record import record_event
from evals.task_state import TaskState

SolverSpec = TypedDict("SolverSpec", {"class": str, "args": Dict[str, Any]})
SolverType = TypeVar("SolverType", bound="Solver")


class SolverResult:
    def __init__(self, output: str, **metadata):
        self._output = output
        self._metadata = metadata

    @property
    def output(self) -> str:
        return self._output

    @property
    def metadata(self) -> dict:
        return self._metadata

    def to_json(self) -> str:
        return json.dumps(
            {
                "output": self.output,
                **self.metadata,
            },
            indent=2,
        )


class Solver(ABC, CompletionFn):
    # We need to inherit from CompletionFn because of how the oaival registry works.

    def __init__(
        self,
        postprocessors: list[str] = [],
        registry: Any = None,
    ) -> None:
        self.postprocessors: list = []
        for postprocessor_path in postprocessors:
            try:
                module_path, class_name = postprocessor_path.rsplit(":", 1)
                module = import_module(module_path)
                postprocessor_class = getattr(module, class_name)
                self.postprocessors.append(postprocessor_class())
            except AttributeError:
                raise ValueError(f"Invalid postprocessor: {postprocessor_path}")

    @abstractmethod
    def _solve(
        self,
        task_state: TaskState,
        **kwargs,
    ) -> SolverResult:
        """
        ARGS
        ====
        `task_state`: A `TaskState` object that contains the task description and the input.
        `kwargs`: Other arguments passed to the solver.

        RETURNS
        =======
        The result of the solver.
        """

    def __call__(
        self,
        task_state: TaskState,
        **kwargs,
    ) -> SolverResult:
        """Deepcopies task_state to prevent solvers from modifying the original object."""
        res = self._solve(deepcopy(task_state), **kwargs)

        if hasattr(self, "postprocessors"):
            # Iteratively apply postprocessors to the output
            for postprocessor in self.postprocessors:
                prev_output = res.output
                res = postprocessor(res)
                record_event(
                    "postprocessor",
                    {
                        "name": postprocessor.__class__.__name__,
                        "input": prev_output,
                        "output": res.output,
                    },
                )
        return res

    @property
    def name(self) -> str:
        """
        Name of the Solver. This is intended mostly for logging.

        RETURNS
        =======
        A human-readable name that describes this solver.
        """
        return type(self).__name__

    @property
    def model_version(self) -> Union[str, dict]:
        """
        Exact version of the underlying model used by the solver

        RETURNS
        =======
        Dictionary mapping name to exact model version. If no models
        are used (e.g. dummy solver) returns empty dictionary
        """
        return {}

    def copy(self: SolverType) -> SolverType:
        #   The deepcopy may be quite heavy for some solvers; if that's the
        #   case they should override this function.
        return deepcopy(self)


class DummySolver(Solver):
    def _solve(
        self,
        task_state: TaskState,
        **kwargs,
    ) -> SolverResult:
        return SolverResult("This is a dummy response.")


class NestedSolver(Solver):
    """An abstract solver class that receives specification of any number of other solvers as an argument."""

    # TODO: Should we allow nested solvers to (also) take Solver classes instead of SolverSpecs?

    def __init__(self, *, postprocessors: list[str] = [], registry=None, **solver_specs):
        super().__init__(postprocessors=postprocessors)
        self.solver_specs = {}
        self._solver_cache = {}

        SolverSpecValidator = TypeAdapter(SolverSpec)
        for name, value in solver_specs.items():
            try:
                SolverSpecValidator.validate_python(value)
                self.solver_specs[name] = value
                self.get_solver(name)  # Initialize the solver
            except ValidationError:
                raise ValueError(f"Expected a sub-solver spec at '{name}', got '{value}'")

        assert (
            self.solver_specs
        ), f"{type(self).__name__} requires at least one sub-solver as an argument"

    def get_solver(self, solver_name: str) -> Solver:
        """
        IMPORTANT: All subclasses of NestedSolver should use this method to reference any
        sub-solvers, otherwise solver copies will not work properly.

        For convenience, your subclass can have a @property method like this:
        ```python
        @property
        def my_sub_solver(self) -> Solver:
            return self.get_solver("my_sub_solver")
        ```
        which is used in the _solve method like this:
        ```python
        def _solve(
            self,
            task_state: TaskState,
            **kwargs,
        ) -> SolverResult:
            ...
            solver_result = self.my_sub_solver(task_state=task_state, **kwargs)
            ...
        ```
        """
        if solver_name not in self._solver_cache:
            solver_spec = self.solver_specs[solver_name]
            self._solver_cache[solver_name] = self._create_solver(solver_spec)
        return self._solver_cache[solver_name]

    def _create_solver(self, solver_spec: SolverSpec) -> Solver:
        module_name, class_name = solver_spec["class"].split(":")
        module = import_module(module_name)
        cls = getattr(module, class_name)
        return cls(**solver_spec["args"])

    def copy(self: SolverType) -> SolverType:
        # The NestedSolver needs to manually copy the sub-solvers, otherwise we will miss any
        # special copy logic they may have.
        solver_copy = deepcopy(self)  # TODO: We should deepcopy without copying the cache
        for name, solver in self._solver_cache.items():
            solver_copy._solver_cache[name] = solver.copy()
        return solver_copy

    @property
    def model_version(self) -> Union[str, dict]:
        """
        Retrieves model versions of each nested solver
        """
        model_versions = {}
        for solver_name, solver in self._solver_cache.items():
            solver_model_version = solver.model_version
            model_versions[solver_name] = solver_model_version

        return model_versions
