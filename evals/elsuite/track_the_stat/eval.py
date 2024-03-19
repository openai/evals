import logging
import random
from typing import Any, Optional

import numpy as np

from evals.elsuite.track_the_stat import prompts, utils
from evals.eval import SolverEval
from evals.record import RecorderBase, record_metrics
from evals.solvers.solver import Solver
from evals.task_state import Message, TaskState

logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class TrackTheStat(SolverEval):
    def __init__(self, task: str, n_samples: Optional[int] = 250, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert task in [
            "median",
            "mode",
        ], f"task must be either 'median' or 'mode', but got {task}"
        self.task = task
        # warn, color in yellow
        logger.warning(
            utils.yellow_string(
                "By nature of what is being evaluated, this eval assumes that the "
                "solver cannot make use of external scratchpads or similar solutions "
                "to explicitly write down the task state at every step. Using solvers "
                "that allow for this functionality will likely produce invalid results."
            )
        )
        self.task_desc = prompts.TASK_DESCRIPTION.format(
            task=task,
            task_further_details=prompts.task_to_further_details[task],
            task_example=prompts.task_to_example[task],
        )
        self.task_fn = utils.task_to_fn[task]
        self.n_samples = n_samples
        self.rng = random.Random(self.seed)

    def eval_sample(self, solver: Solver, sample: Any, rng: random.Random) -> None:
        capped_inf_list = np.random.default_rng(sample["seed"]).integers(0, 100, size=300)
        metrics = self._eval_sample(solver, capped_inf_list)

        record_metrics(**metrics)

    def _eval_sample(self, solver: Solver, capped_inf_list: list[int]) -> dict:
        violation = False
        task_state = TaskState(task_description=self.task_desc, messages=[])
        for i, num in enumerate(capped_inf_list):
            curr_list = capped_inf_list[: i + 1]
            task_state.messages.append(Message(role="user", content=str(num)))
            task_state.current_state = utils.compute_state(curr_list, self.task)
            solver_output = solver(task_state).output
            solver_response = utils.parse_solver_output(solver_output, self.task)
            if solver_response is None:
                violation = True
                break
            if round(solver_response, 1) != round(self.task_fn(curr_list), 1):
                break
            task_state.messages.append(Message(role="assistant", content=solver_output))

        return {
            "max_length": len(curr_list) - 1,
            "violation": violation,
        }

    def run(self, recorder: RecorderBase):
        samples = self._get_samples()
        self.eval_all_samples(recorder, samples)
        logged_metrics: list[dict] = recorder.get_metrics()

        agg_metrics = self._compute_agg_metrics(logged_metrics)
        return agg_metrics

    def _compute_agg_metrics(self, logged_metrics: list[dict]) -> dict:
        max_lengths = np.array([x["max_length"] for x in logged_metrics])

        agg_metrics = {
            "avg_max_length": np.mean(max_lengths),
            "stddev_max_length": np.std(max_lengths),
            "median_max_length": np.median(max_lengths),
            "max_max_length": np.max(max_lengths),
            "min_max_length": np.min(max_lengths),
            "violation_rate": np.mean([x["violation"] for x in logged_metrics]),
        }
        # necessary for serialization, json doesn't like np floats
        agg_metrics = {k: float(v) for k, v in agg_metrics.items()}
        return agg_metrics

    def _get_samples(self) -> list[dict]:
        samples = [{"seed": num} for num in range(self.n_samples)]
        samples = self.rng.sample(samples, self.n_samples)  # shuffling
        return samples
