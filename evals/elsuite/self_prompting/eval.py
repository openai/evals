import json
import logging
import random
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

import evals
import evals.metrics
from evals.api import CompletionFn
from evals.elsuite.self_prompting.task_description import sample_in_token, task_description_template
from evals.eval import SolverEval
from evals.registry import registry
from evals.solvers.solver import Solver
from evals.task_state import TaskState
from evals.utils.log_utils import extract_final_results, extract_spec

logger = logging.getLogger(__name__)


class SelfPrompting(SolverEval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        samples_jsonl: str,
        tasker_models: list[str],
        n_tasks: int = 50,
        n_samples_per_task: int = 10,
        n_preview_samples: int = 5,
        baseline_logpath: Optional[str] = None,
        *args,
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        # CI doesn't have access to model APIs, so replace tasker_models with dummy models
        # if we're running in CI (i.e. if the first completion_fn is a DummyCompletionFn)
        if isinstance(completion_fns[0], evals.api.DummyCompletionFn):
            tasker_models = ["dummy" for _ in tasker_models]

        self.samples_jsonl = samples_jsonl
        self.tasker_models = tasker_models
        self.n_tasks = n_tasks
        self.n_samples_per_task = n_samples_per_task
        self.n_preview_samples = n_preview_samples
        self.baseline_logpath = (
            self._prefix_registry_path(baseline_logpath) if baseline_logpath else None
        )
        assert len(self.tasker_models) > 0, "Must provide at least one tasker model"
        assert self.n_tasks > 0, "Must provide at least one task"
        assert self.n_samples_per_task > 0, "Must provide at least one sample per task"

        np.random.seed(self.seed)

        self.tasker_completion_fns = {}
        for tasker_model in self.tasker_models:
            self.tasker_completion_fns[tasker_model] = registry.make_completion_fn(tasker_model)

    def eval_sample(self, solver: Solver, sample: Any, rng: random.Random):
        if sample["stage"] == "prompting":
            return self._run_prompting(solver, sample)
        elif sample["stage"] == "tasking":
            return self._run_tasking(sample)
        else:
            raise ValueError(f"Invalid stage {sample['stage']}")

    def _run_prompting(self, solver: Solver, sample: Any, *_):
        # Prompt the prompter_model to generate a prompt for the tasker_model
        task_description = task_description_template.format(
            instruction=sample["task"]["instruction"],
            samples=json.dumps(sample["task"]["train_samples"], indent=2),
            tasker_model=sample["tasker_model"],
        )
        task_state = TaskState(
            task_description=task_description,
            current_state={
                "instruction": sample["task"]["instruction"],
                "samples": sample["task"]["train_samples"],
                "tasker_model": sample["tasker_model"],
            },
        )
        solver_result = solver(task_state)
        model_instruction = solver_result.output

        prompt_rule_violation = sample_in_token not in model_instruction

        output = {
            **sample,
            "task_description": task_description,
            "current_state": task_state.current_state,
            "prompting_solver_metadata": solver_result.to_json(),
            "model_instruction": model_instruction,
            "prompt_rule_violation": prompt_rule_violation,
        }
        return output

    def _run_tasking(self, sample: Any, *_):
        tasker_completion_fn = self.tasker_completion_fns[sample["tasker_model"]]

        if sample_in_token in sample["model_instruction"]:
            # Fill in the sample input
            full_prompt = sample["model_instruction"].replace(sample_in_token, sample["input"])
        else:
            # Append the sample input
            full_prompt = f"{sample['model_instruction']}\n{sample['input']}"
        tasker_output = tasker_completion_fn(full_prompt).get_completions()[0]

        exact = 1 if tasker_output == sample["output"] else 0
        fuzzy = 1 if tasker_output in sample["output"] or sample["output"] in tasker_output else 0

        output = {
            **sample,
            "full_prompt": full_prompt,
            "tasker_output": tasker_output,
            "exact": exact,
            "fuzzy": fuzzy,
        }
        evals.record.record_metrics(**output)
        return output

    def _calculate_improvement_wrt_baseline(
        self, current_res: dict[str, float]
    ) -> dict[str, float]:
        if self.baseline_logpath is None:
            logger.warn("SKIPPING IMPROVEMENT METRICS. (No baseline logpath provided.)")
            return {}

        # Check that baseline was run on the same tasker models, tasks, and samples
        baseline_spec = extract_spec(Path(self.baseline_logpath))
        try:
            spec_args = baseline_spec["run_config"]["eval_spec"]["args"]
        except KeyError:
            logger.warn("SKIPPING IMPROVEMENT METRICS. (Failed to validate baseline spec.)")
            return {}
        if set(spec_args["tasker_models"]) != set(self.tasker_models):
            logger.warn(
                f"SKIPPING IMPROVEMENT METRICS. (Baseline tasker_models {spec_args['tasker_models']} do not match {self.tasker_models}.)"
            )
            return {}
        if (
            spec_args["n_tasks"] != self.n_tasks
        ):  # TODO: Ideally we would check that the tasks are the same
            logger.warn(
                f"SKIPPING IMPROVEMENT METRICS. (Baseline n_tasks {spec_args['n_tasks']} does not match {self.n_tasks}.)"
            )
            return {}
        if spec_args["n_samples_per_task"] != self.n_samples_per_task:
            logger.warn(
                f"SKIPPING IMPROVEMENT METRICS. (Baseline n_samples_per_task {spec_args['n_samples_per_task']} does not match {self.n_samples_per_task}.)"
            )
            return {}

        baseline_res = extract_final_results(Path(self.baseline_logpath))

        def normalized_improvement(current, baseline):
            """
            Returns a score between -1 and 1, where
            -1 means the current score maximally regresses from the baseline (i.e. the current score is 0)
            0 means the current score is the same as the baseline
            +1 means the current score achieves max improvement over the baseline
            """
            if current < baseline:
                return (current - baseline) / baseline
            else:
                return (current - baseline) / (1 - baseline)

        improvement_scores = {
            "accuracy_improvement_wrt_oriprompt": normalized_improvement(
                current_res["accuracy"], baseline_res["accuracy"]
            ),
            "accuracy_fuzzy_improvement_wrt_oriprompt": normalized_improvement(
                current_res["accuracy_fuzzy"], baseline_res["accuracy_fuzzy"]
            ),
            "baseline_accuracy": baseline_res["accuracy"],
            "baseline_accuracy_fuzzy": baseline_res["accuracy_fuzzy"],
        }
        logger.info(f"Improvement scores: {improvement_scores}")
        return improvement_scores

    def _run_impl(self, recorder: evals.record.Recorder) -> dict[str, Union[float, int]]:
        samples = self.get_samples()

        # Shuffle and limit samples
        np.random.shuffle(samples)
        samples_by_task = samples[: self.n_tasks]
        assert len(samples_by_task) == self.n_tasks
        for task in samples_by_task:
            np.random.shuffle(task["test_samples"])
            np.random.shuffle(task["train_samples"])
            task["test_samples"] = task["test_samples"][: self.n_samples_per_task]
            task["train_samples"] = task["train_samples"][: self.n_preview_samples]
            assert len(task["test_samples"]) == self.n_samples_per_task
            assert len(task["train_samples"]) == self.n_preview_samples

        # Run prompting
        prompting_samples = []
        for task in samples_by_task:
            for tasker_model in self.tasker_models:
                prompting_samples.append(
                    {
                        "stage": "prompting",
                        "tasker_model": tasker_model,
                        "task": task,
                    }
                )
        assert len(prompting_samples) == len(self.tasker_models) * self.n_tasks
        prompting_results = self.eval_all_samples(recorder, prompting_samples)

        # Run tasking
        tasking_samples = []  # Store in flattened list for parallel eval
        for prompt_res in prompting_results:
            prompt_res["stage"] = "tasking"  # Update stage
            for sample in prompt_res["task"]["test_samples"]:
                tasking_samples.append(
                    {
                        **prompt_res,
                        "input": sample["input"],
                        "output": sample["output"],
                    }
                )
        assert len(tasking_samples) == len(prompting_results) * self.n_samples_per_task
        self.eval_all_samples(recorder, tasking_samples)

        # The score of a Prompter is the average score of all Tasker models it writes prompts for
        metrics = recorder.get_metrics()

        # Primary metrics
        result = {
            "accuracy": np.mean([metric["exact"] for metric in metrics]),
            "accuracy_fuzzy": np.mean([metric["fuzzy"] for metric in metrics]),
        }
        # Relative improvement against baseline
        improvement_scores = self._calculate_improvement_wrt_baseline(result)
        if improvement_scores:
            result.update(improvement_scores)

        # Peripheral metrics
        result.update(
            {
                "prompt_rule_violation_rate": np.mean(
                    [int(metric["prompt_rule_violation"]) for metric in metrics]
                ),
                "n_samples": len(metrics),
            }
        )

        # Breakdown by tasker model
        def compute_mean_tasker(key, tasker_model):
            return np.mean(
                [metric[key] for metric in metrics if metric["tasker_model"] == tasker_model]
            )

        for tasker in self.tasker_models:
            result.update(
                {
                    f"accuracy_{tasker}": compute_mean_tasker("exact", tasker),
                    f"accuracy_fuzzy_{tasker}": compute_mean_tasker("fuzzy", tasker),
                }
            )

        return result
