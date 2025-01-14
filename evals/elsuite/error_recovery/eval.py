import copy
import random
from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Sequence

import evals
import evals.metrics
import evals.record
from evals.api import CompletionFn
from evals.elsuite.error_recovery.defaults import (
    DEFAULT_FINAL_ANSWER_MESSAGE,
    DEFAULT_MISTAKE_MESSAGE,
    DEFAULT_TASK_DESCRIPTION,
    TASK_SPECIFIC_EXTRACTION_INFO,
)
from evals.eval import SolverEval
from evals.solvers.solver import Solver
from evals.task_state import Message, TaskState

# possible Mistake NOTIFiciation POSitions
MistakeNotifPos = Literal["immediate", "end"]


@dataclass
class Sample:
    question: str
    correct_steps: Sequence[str]
    incorrect_step: str
    target: Any
    task: str
    num_ground_truth_steps: int
    mistake_index: int


class ErrorRecovery(SolverEval):
    def __init__(
        self,
        completion_fns: Sequence[CompletionFn],
        samples_jsonl: str,
        n_samples: Optional[int] = None,
        mistake_notification_position: Optional[MistakeNotifPos] = None,
        mistake_notification_for_ir_only: bool = False,
        mark_as_own_reasoning: bool = True,
        final_answer_prompt_role: str = "system",
        *args,
        **kwargs,
    ):
        """Evaluate a solver on the error recovery task.

        Args:
            completion_fns: The completion functions to evaluate. (should be a single solver)
            samples_jsonl: The relative path to the samples jsonl file in evals/registry/data.
            n_samples: The number of samples to use. If None, use all samples.
            mistake_notification_position: The position of the mistake
                notification. Options are "immediate" for right after the provided
                reasoning, or "end" for right after the model-generated reasoning.
                If None, no mistake notification is added.
            mistake_notification_for_ir_only: Whether to only add the mistake notification
                for the incorrect reasoning case. If True, the mistake notification is
                added for the incorrect reasoning case, and not for the correct reasoning
                or no reasoning cases.
            mark_as_own_reasoning: Whether to include the sample reasoning as an
                'assistant' or 'user' message.
            final_answer_prompt_role: The role to use for the final answer prompt. Should
                be either "system" or "user".
        """
        super().__init__(
            completion_fns=completion_fns, samples_jsonl=samples_jsonl, *args, **kwargs
        )

        self.n_samples = n_samples
        self.mistake_notif_pos: Optional[MistakeNotifPos] = mistake_notification_position
        self.mistake_notif_ir_only = mistake_notification_for_ir_only

        # there are some issues with passing bools in from extra_eval_params
        assert isinstance(mark_as_own_reasoning, bool)
        self.mark_as_own_reasoning = mark_as_own_reasoning

        self.final_answer_prompt_role = final_answer_prompt_role
        assert self.final_answer_prompt_role in ["system", "user"]

    def eval_sample(self, solver: Solver, sample: Sample, rng: random.Random, extra_logging=None):
        task = sample.task

        # Get the baseline with no provided reasoning
        nr_task_state = self._get_no_reasoning_task_state(sample)
        # only "end" makes sense for 'no reasoning'
        nr_notif_pos = "end" if self.mistake_notif_pos == "end" else None
        if self.mistake_notif_ir_only:
            nr_notif_pos = None

        nr_answer = self._get_answer(
            solver=solver,
            task_state=nr_task_state,
            sample=sample,
            mistake_notif_pos=nr_notif_pos,
        )

        # Run with correct reasoning
        cr_task_state = self._get_correct_reasoning_task_state(sample)
        cr_notif_pos = self.mistake_notif_pos
        if self.mistake_notif_ir_only:
            cr_notif_pos = None

        cr_answer = self._get_answer(
            solver=solver,
            task_state=cr_task_state,
            sample=sample,
            mistake_notif_pos=cr_notif_pos,
        )

        # Run with incorrect reasoning
        ir_task_state = self._get_incorrect_reasoning_task_state(sample)
        ir_notif_pos = self.mistake_notif_pos

        ir_answer = self._get_answer(
            solver=solver,
            task_state=ir_task_state,
            sample=sample,
            mistake_notif_pos=ir_notif_pos,
        )

        assert len(sample.correct_steps) == sample.mistake_index

        metrics = {
            "task": task,
            "num_ground_truth_steps": sample.num_ground_truth_steps,
            "mistake_index": sample.mistake_index,
            "target": str(sample.target),  # ground truth answer
            "mistake_notification_position": self.mistake_notif_pos,
            "mistake_notification_for_ir_only": self.mistake_notif_ir_only,
            "NR_sampled": nr_answer,
            "CR_sampled": cr_answer,
            "IR_sampled": ir_answer,
            "NR_correct": nr_answer == str(sample.target),
            "CR_correct": cr_answer == str(sample.target),
            "IR_correct": ir_answer == str(sample.target),
        }
        evals.record.record_metrics(**metrics)

    def _get_no_reasoning_task_state(self, sample: Sample) -> TaskState:
        task_description = DEFAULT_TASK_DESCRIPTION
        no_reasoning_messages = [
            Message(role="user", content=sample.question),
        ]
        no_reasoning_task_state = TaskState(
            task_description=task_description,
            messages=no_reasoning_messages,
        )
        return no_reasoning_task_state

    def _get_correct_reasoning_task_state(self, sample: Sample) -> TaskState:
        task_description = DEFAULT_TASK_DESCRIPTION
        correct_steps = "\n".join(sample.correct_steps)
        reasoning_role = "assistant" if self.mark_as_own_reasoning else "user"
        correct_reasoning_messages = [
            Message(role="user", content=sample.question),
            Message(role=reasoning_role, content=correct_steps),
        ]
        correct_reasoning_task_state = TaskState(
            task_description=task_description,
            messages=correct_reasoning_messages,
        )
        return correct_reasoning_task_state

    def _get_incorrect_reasoning_task_state(
        self,
        sample: Sample,
    ) -> TaskState:
        task_description = DEFAULT_TASK_DESCRIPTION
        correct_steps = "\n".join(sample.correct_steps)
        steps_with_incorrect_reasoning = f"{correct_steps}\n{sample.incorrect_step}"
        reasoning_role = "assistant" if self.mark_as_own_reasoning else "user"
        incorrect_reasoning_messages = [
            Message(role="user", content=sample.question),
            Message(role=reasoning_role, content=steps_with_incorrect_reasoning),
        ]

        incorrect_reasoning_task_state = TaskState(
            task_description=task_description,
            messages=incorrect_reasoning_messages,
        )
        return incorrect_reasoning_task_state

    def _get_answer(
        self,
        solver: Solver,
        task_state: TaskState,
        sample: Sample,
        mistake_notif_pos: Optional[MistakeNotifPos],
    ) -> str:
        """Get a final answer from the solver for a given sample.

        Args:
            solver: The solver to use.
            task_state: The task state to use.
            sample: The Sample being evaluated (relevant for answer extraction).
            mistake_notification_position: The position of the mistake notification.
                Options are "immediate" for right after the provided reasoning, or "end" for right
                after the model-generated reasoning. If None, no mistake notification is added.

        TODO (ian): Work out whether to add mistake notification to 'no reasoning' baseline
        """
        mistake_message = Message("user", DEFAULT_MISTAKE_MESSAGE)
        if mistake_notif_pos == "immediate":
            task_state.messages.append(mistake_message)

        output = solver(task_state=task_state).output
        task_state.messages.append(Message("assistant", output))

        # run solver again if mistake notification is at the end
        if mistake_notif_pos == "end":
            task_state.messages.append(mistake_message)
            output = solver(task_state=task_state).output
            task_state.messages.append(Message("assistant", output))

        answer = self._extract_final_answer(solver=solver, task_state=task_state, sample=sample)
        return answer

    def _run_impl(self, recorder: evals.record.Recorder):
        samples = self.get_samples()

        self.eval_all_samples(recorder, samples)
        metrics = recorder.get_metrics()

        NR_correct_rate = len([i for i in metrics if i["NR_correct"]]) / len(metrics)
        CR_correct_rate = len([i for i in metrics if i["CR_correct"]]) / len(metrics)
        IR_correct_rate = len([i for i in metrics if i["IR_correct"]]) / len(metrics)

        results = {
            "NR_correct_rate": NR_correct_rate,
            "CR_correct_rate": CR_correct_rate,
            "IR_correct_rate": IR_correct_rate,
        }

        # Split results per type of task
        all_tasks = set([i["task"] for i in metrics])
        for task in all_tasks:
            filtered_metrics = [i for i in metrics if i["task"] == task]
            NR_correct_rate = len([i for i in filtered_metrics if i["NR_correct"]]) / len(
                filtered_metrics
            )
            CR_correct_rate = len([i for i in filtered_metrics if i["CR_correct"]]) / len(
                filtered_metrics
            )
            IR_correct_rate = len([i for i in filtered_metrics if i["IR_correct"]]) / len(
                filtered_metrics
            )

            # we use hyphens in the task name so they can be extracted by splitting on underscores
            task_string = task.replace("_", "-")
            results.update(
                {
                    f"task_{task_string}_NR_correct_rate": NR_correct_rate,
                    f"task_{task_string}_CR_correct_rate": CR_correct_rate,
                    f"task_{task_string}_IR_correct_rate": IR_correct_rate,
                }
            )

        return results

    def _extract_final_answer(self, solver: Solver, task_state: TaskState, sample: Sample):
        """Extract the final answer from the solver output using the same solver."""
        task_state = copy.deepcopy(task_state)

        task_specific_info = TASK_SPECIFIC_EXTRACTION_INFO[sample.task]
        final_answer_prompt = DEFAULT_FINAL_ANSWER_MESSAGE + task_specific_info

        task_state.messages.append(
            Message(role=self.final_answer_prompt_role, content=final_answer_prompt)
        )
        answer = solver(task_state=task_state).output

        return answer

    def get_samples(self) -> List[Sample]:
        samples = super().get_samples()

        if self.n_samples is not None:
            assert (
                len(samples) >= self.n_samples
            ), f"Can't get {self.n_samples} samples from a dataset with {len(samples)} samples"
            samples = samples[: self.n_samples]
        return [Sample(**sample_dict) for sample_dict in samples]
