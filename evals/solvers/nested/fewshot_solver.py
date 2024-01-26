import random
from typing import Any

from evals.data import get_jsonl
from evals.solvers.solver import NestedSolver, Solver, SolverResult, SolverSpec
from evals.task_state import Message, TaskState


class FewShotSolver(NestedSolver):
    def __init__(
        self,
        train_jsonl: str,  # TODO: move this to be handled eval-side
        n_shots: int,
        base_solver: SolverSpec,
        repeat_task_description: bool = False,
        registry: Any = None,
        seed: int = 121123,
    ):
        super().__init__(registry=registry, base_solver=base_solver)
        self.n_shots = n_shots
        self.repeat_task_description = repeat_task_description
        self.rng = random.Random(seed)

        train_data = get_jsonl(train_jsonl)

        assert (
            len(train_data) >= n_shots
        ), f"Insufficient training data provided for few-shot solver, provide at least {n_shots} samples. Size of training data: {len(train_data)}"
        assert (
            isinstance(train_data[0], list)
            or isinstance(train_data[0], dict)
            and "input" in train_data[0]
            and "ideal" in train_data[0]
        ), "Incorrect format of training data provided for few-shot solver, each data point should be a list of messages or a dictionary with 'input' and 'ideal' keys."

        formatted_train_data = []

        if isinstance(train_data[0], dict):
            if "content" in train_data[0]["input"][0]:
                for datapoint in train_data:
                    formatted_train_data += [
                        (
                            Message(role="user", content=datapoint["input"][0]["content"]),
                            Message(role="assistant", content=datapoint["ideal"]),
                        )
                    ]
            else:
                for datapoint in train_data:
                    formatted_train_data += [
                        (
                            Message(role="user", content=datapoint["input"]),
                            Message(role="assistant", content=datapoint["ideal"]),
                        )
                    ]
        elif isinstance(train_data[0], list):
            formatted_train_data = [
                (
                    Message(role=msg_list[0]["role"], content=msg_list[0]["content"]),
                    Message(role=msg_list[1]["role"], content=msg_list[1]["content"]),
                )
                for msg_list in train_data
            ]
        else:
            raise ValueError(
                f"Unknown format of training data provided for few-shot solver, each data point should be a list of messages or a dictionary with 'input' and 'ideal' keys. Example data point: {train_data[0]}"
            )

        self.train_data = formatted_train_data

    @property
    def base_solver(self) -> Solver:
        return self.get_solver("base_solver")

    def _solve(
        self,
        task_state: TaskState,
        **kwargs,
    ) -> SolverResult:
        new_task_state = self._modify_task_state(task_state)
        return self.base_solver(new_task_state)

    def _modify_task_state(self, task_state: TaskState) -> TaskState:
        assert all(
            user_message not in task_state.messages
            for (user_message, assistant_message) in self.train_data
        ), f"The few-shot training data provided contains the current test set point: {task_state.messages}. Check datasets for contamination."

        # Sample n_shots from train samples
        samples = self.rng.sample(self.train_data, self.n_shots)

        msgs = []
        for idx, (user_message, assistant_message) in enumerate(samples):
            if idx != 0 and self.repeat_task_description:
                msgs.append(Message(role="system", content=task_state.task_description))

            msgs += [user_message, assistant_message]

        # Context for current sample
        msgs += task_state.messages

        return TaskState(
            task_description=task_state.task_description,
            messages=msgs,
            current_state=task_state.current_state,
        )

    @property
    def name(self) -> str:
        return self.base_solver.name
