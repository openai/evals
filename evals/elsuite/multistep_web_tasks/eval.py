import logging
from typing import Any

import docker

import evals
from evals.api import CompletionFn
from evals.elsuite.multistep_web_tasks.constants import DOCKER_CLIENT_TIMEOUT
from evals.elsuite.multistep_web_tasks.session import Session
from evals.elsuite.multistep_web_tasks.utils import load_experiment_config_from_dict
from evals.elsuite.multistep_web_tasks.webarena.core.env import ExperimentResult
from evals.elsuite.multistep_web_tasks.webarena.eval_run import run_experiment
from evals.eval import SolverEval
from evals.record import RecorderBase
from evals.solvers.solver import Solver

logger = logging.getLogger(__name__)


class MultistepWebTasks(SolverEval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        *args,
        samples_jsonl: str = "tasks.jsonl",
        **kwargs,
    ):
        super().__init__(
            completion_fns=completion_fns,
            samples_jsonl=samples_jsonl,
            *args,
            **kwargs,
        )
        assert len(completion_fns) == 1, "Only one completion fn supported"
        docker_client = docker.from_env(timeout=DOCKER_CLIENT_TIMEOUT)
        self.session = Session(docker_client)

    def eval_sample(self, solver: Solver, sample: dict, rng: Any) -> None:
        experiment_config = load_experiment_config_from_dict(sample)

        result: ExperimentResult = run_experiment(solver, experiment_config, self.session)

        evals.record.record_metrics(  # type: ignore (always broken)
            task_id=sample["task_id"],
            score=result.score,
            final_action=result.trajectory[-1].action.parsed_prediction,  # type: ignore (should never be None)
            trajectory_length=len(result.trajectory),
        )

    def run(self, recorder: RecorderBase):
        samples = self.get_samples()
        self.session.add_samples(samples)
        # with statement handles setting up docker containers and tearing them down on completion/error
        with self.session:
            self.eval_all_samples(recorder, samples)
            metrics = recorder.get_metrics()

        return {
            "scores": {m["task_id"]: m["score"] for m in metrics},
            "final_actions": {m["task_id"]: m["final_action"] for m in metrics},
            "trajectory_lengths": {m["task_id"]: m["trajectory_length"] for m in metrics},
        }
