import os
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory

import numpy as np

from evals.api import CompletionFn
from evals.elsuite.hr_ml_agent_bench.autoeval import run as run_auto_eval
from evals.elsuite.hr_ml_agent_bench.utils import is_gpu_available
from evals.eval import SolverEval
from evals.record import Recorder, record_metrics
from evals.registry import Registry
from evals.solvers.solver import Solver

registry = Registry()
logger = getLogger(__name__)


@dataclass(frozen=True)
class Sample:
    task_name: str
    research_problem: str
    max_steps: int
    max_time: int
    max_seconds_per_step: int
    requires_gpu: bool = False

    def __post_init__(self):
        assert (
            isinstance(self.task_name, str) and self.task_name != ""
        ), "`task_name` must be a non-empty string."

        assert (
            isinstance(self.research_problem, str) and self.research_problem != ""
        ), "`research_problem` must be a non-empty string."

        assert (
            isinstance(self.max_steps, int) and self.max_steps > 0
        ), "`max_steps` must be positive."

        assert isinstance(self.max_time, int) and self.max_time > 0, "`max_time` must be positive."

        assert (
            isinstance(self.max_seconds_per_step, int) and self.max_seconds_per_step > 0
        ), "`max_seconds_per_step` must be positive."


class MLAgentBench(SolverEval):
    def __init__(self, completion_fns: list[CompletionFn], *args, **kwargs):
        super().__init__(completion_fns, *args, **kwargs)

        if not in_ci() and os.getenv("EVALS_SEQUENTIAL") not in {"1", "yes", "true"}:
            raise ValueError(
                "Multi-threading not supported! Please set the environment variable "
                "`EVALS_SEQUENTIAL` to 1."
            )

    def eval_sample(self, solver: Solver, raw_sample: dict, rng: Random) -> None:
        del rng

        sample = Sample(**raw_sample)

        if sample.requires_gpu and not is_gpu_available():
            logger.warning(
                f"Warning: you are attempting to run the GPU-variant of the `{sample.task_name}` "
                f"task, but no GPU was found! To run the CPU-variant of `{sample.task_name}`, "
                f"use the task ID `hr-ml-agent-bench.{sample.task_name.replace('_', '-')}.cpu.v0`."
            )

        with TemporaryDirectory() as tmpdir:
            result = run_auto_eval(
                solver=solver,
                log_dir=Path(tmpdir) / "logs",
                work_dir=Path(tmpdir) / "workspace",
                task_name=sample.task_name,
                research_problem=sample.research_problem,
                max_steps=sample.max_steps,
                max_time=sample.max_time,
                max_seconds_per_step=sample.max_seconds_per_step,
            )

        record_metrics(
            task_name=sample.task_name,
            # Raw scores in the original unit of the task.
            model_score=result.model_score,
            naive_baseline_score=result.naive_baseline_score,
            human_baseline_score=result.human_baseline_score,
            # Normalized scores are in the range [0, 1] where higher is better.
            model_score_normalized=result.model_score_normalized,
            naive_baseline_score_normalized=result.naive_baseline_score_normalized,
            human_baseline_score_normalized=result.human_baseline_score_normalized,
            # Human-relative scores are in the range [0, 1] where 0 is the naive
            # baseline and 1 is the human baseline.
            model_score_humanrelative=result.model_score_humanrelative,
        )

    def _run_impl(self, recorder: Recorder) -> dict:
        samples = self.get_samples()
        self.eval_all_samples(recorder, samples)
        metrics = recorder.get_metrics()

        final_report = {}

        for metric in metrics:
            task_metrics = {k: v for k, v in metric.items()}
            final_report.update(task_metrics)

        if metrics:
            final_report["avg_humanrelative_score"] = np.mean(
                [d["model_score_humanrelative"] for d in metrics]
            )

        return final_report


def in_ci():
    return os.environ.get("GITHUB_ACTIONS") == "true"
