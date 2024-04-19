import logging
import os
from typing import Any, Optional

import numpy as np

import evals
import evals.metrics
from evals.api import CompletionFn
from evals.eval import SolverEval
from evals.solvers.solver import Solver
from evals.task_state import Message, TaskState


class MatchWithSolvers(SolverEval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        samples_jsonl: str,
        task_description: str,
        n_samples: Optional[int] = None,
        shuffle: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert (
            len(task_description) > 0
        ), "Must provide a task description or a path to a .txt file containing one."

        if os.path.exists(task_description):
            self.task_description = open(task_description, "r").read()
            logging.info(f"Loaded task description from {task_description}")
        else:
            self.task_description = task_description

        self.samples_jsonl = samples_jsonl
        self.n_samples = n_samples
        self.shuffle = shuffle
        np.random.seed(self.seed)

    def eval_sample(self, solver: Solver, sample: Any, *_):
        assert isinstance(sample, dict), "sample must be a dict"
        assert "input" in sample, "sample must have an 'input' key"
        assert "ideal" in sample, "sample must have an 'ideal' key"
        assert isinstance(sample["ideal"], str) or isinstance(
            sample["ideal"], list
        ), "sample['ideal'] must be a string or list of strings"

        messages = [Message(**msg) for msg in sample["input"]]

        task_state = TaskState(
            task_description=self.task_description,
            messages=messages,
        )

        solver_result = solver(task_state)
        output = solver_result._output

        ideal = sample["ideal"] if isinstance(sample["ideal"], str) else sample["ideal"][0]

        return evals.record_and_check_match(
            prompt=sample["input"],
            sampled=output,
            expected=[ideal, ideal.capitalize()],
        )

    def _run_impl(self, recorder):
        samples = self.get_samples()

        if self.shuffle:
            np.random.shuffle(samples)
        samples = samples[: self.n_samples] if self.n_samples is not None else samples
        self.eval_all_samples(recorder, samples)

        events = recorder.get_events("match")
        return {
            "accuracy": evals.metrics.get_accuracy(events),
            "bootstrap_std": evals.metrics.get_bootstrap_accuracy_std(events),
        }
