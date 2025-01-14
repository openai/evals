import random
from typing import Any, Dict, Sequence

import evals
import evals.metrics
from evals.api import CompletionFn
from evals.elsuite.sandbagging.defaults import DEFAULT_MMLU_MSG
from evals.elsuite.sandbagging.utils import record_mmlu_sample
from evals.eval import SolverEval
from evals.solvers.solver import Solver
from evals.task_state import Message, TaskState


class MMLU(SolverEval):
    def __init__(
        self,
        completion_fns: Sequence[CompletionFn],
        samples_jsonl: str,
        *args,
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)

        self.completion_fns = completion_fns
        self.samples_jsonl = samples_jsonl

    def eval_sample(
        self, solver: Solver, sample: Dict[str, Any], rng: random.Random, extra_logging=None
    ):
        del rng
        assert isinstance(sample, Dict), f"sample must be dict but is {type(sample)}"
        assert isinstance(
            sample["question"], str
        ), f'"question" key in sample must be string but is {type(sample["question"])}'
        assert isinstance(
            sample["answer"], str
        ), f'"answer" key in sample must be string but is {type(sample["answer"])}'

        question, answer = sample["question"], sample["answer"]

        task_description = DEFAULT_MMLU_MSG
        question_msg = Message(role="user", content=question)

        task_state = TaskState(
            task_description=task_description,
            messages=[question_msg],
            current_state=None,
        )
        result = solver(task_state=task_state)
        output = result.output
        output = output.lstrip()

        prompt = [question_msg]
        if "prompt" in result.metadata:
            prompt = result.metadata["prompt"]

        return record_mmlu_sample(
            prompt=prompt,
            sampled=output,
            expected=answer,
            extra_logging=extra_logging,
        )

    def _run_impl(self, recorder: evals.record.Recorder):
        samples = self.get_samples()

        self.eval_all_samples(recorder, samples)
        events = recorder.get_events("metrics")

        return {
            "accuracy": evals.metrics.get_accuracy(events),
            "bootstrap_std_target": evals.metrics.get_bootstrap_accuracy_std(events),
        }
