import random
from collections import deque
from typing import Any, Deque, Optional

import numpy as np

from evals.elsuite.already_said_that import distractors, prompts, utils
from evals.eval import SolverEval
from evals.record import RecorderBase, record_metrics
from evals.solvers.solver import Solver
from evals.task_state import Message, TaskState


class AlreadySaidThat(SolverEval):
    def __init__(
        self,
        distractor_variant: str,
        adversarial: bool = True,
        max_turns: int = 100,
        n_samples: Optional[int] = 250,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.distractor_variant = distractor_variant
        self.distractor_data = distractors.get_distractors(self.distractor_variant)
        distractor_example = self.distractor_data.pop()
        distractor_word = distractors.get_distractor_word(distractor_example.question)
        self.task_description = prompts.TASK_DESCRIPTION.format(
            distractor_question=distractor_example.question,
            distractor_answer=distractor_example.ideal,
            distractor_word=distractor_word,
        )
        self.num_distractors = len(self.distractor_data)
        self.max_turns = max_turns
        self.adversarial = adversarial
        self.n_samples = n_samples
        self.rng = random.Random(self.seed)

    def eval_sample(self, solver: Solver, sample: dict, rng: random.Random) -> None:
        words = sample["words"]
        # make a deque of the (shuffled) distractor data, will be faster to rotate
        distractor_data = deque(rng.sample(self.distractor_data, k=self.num_distractors))

        conversation_metrics = self._conversation_loop(solver, words, distractor_data, rng)

        record_metrics(**conversation_metrics)

    def _conversation_loop(
        self,
        solver: Solver,
        words: list[str],
        distractor_data: Deque[dict[str, str]],
        rng,
    ) -> dict[str, Any]:
        convo_metrics = {
            "num_distractors": 0,
            "num_turns": 0,
            "was_false_pos": False,
            "was_false_neg": False,
            "violation_occurred": False,
            "distractor_accuracy": np.nan,
        }

        words_prev_shown = set()
        words_not_shown = set(words)
        words_from_solver = set()
        words_from_distractors = set()

        distractor_correctness = []

        task_state = TaskState(task_description=self.task_description)

        while convo_metrics["num_turns"] < self.max_turns:
            # conversation
            distracting_words = (
                words_from_solver.union(words_from_distractors) if self.adversarial else set()
            )
            message, message_words, distractor_added = utils.build_message(
                words_not_shown=words_not_shown,
                words_prev_shown=words_prev_shown,
                distracting_words=distracting_words,
                rng=rng,
                distractor_sample=distractor_data[0] if distractor_data else None,
            )
            task_state.messages.append(message)
            solver_output = solver(task_state).output
            task_state.messages.append(Message(role="assistant", content=solver_output))

            # track performance
            parsing_results = utils.parse_solver_output(
                solver_output, message_words, words_prev_shown, distractor_added
            )
            convo_metrics["violation_occurred"] = parsing_results["violation_occurred"]
            mistake_made = parsing_results["mistake_made"]
            if distractor_added is not None:
                distractor_correctness.append(not mistake_made)
                convo_metrics["num_distractors"] += 1
                words_from_distractors.update(message_words)
                # move the distractor we just used to the end of the queue
                distractor_data.rotate(-1)
            elif convo_metrics["violation_occurred"] or (mistake_made and distractor_added is None):
                convo_metrics["was_false_pos"] = parsing_results["false_positive"]
                convo_metrics["was_false_neg"] = parsing_results["false_negative"]
                break
            else:
                words_prev_shown.update(message_words)
                words_not_shown.difference_update(message_words)
            words_from_solver.update(parsing_results["solver_words"])
            convo_metrics["num_turns"] += 1

        convo_metrics["distractor_accuracy"] = (
            np.mean(distractor_correctness) if distractor_correctness else np.nan
        )

        return convo_metrics

    def run(self, recorder: RecorderBase):
        samples = self._get_samples()
        self.eval_all_samples(recorder, samples)
        logged_metrics: list[dict] = recorder.get_metrics()

        agg_metrics = self._compute_agg_metrics(logged_metrics)
        return agg_metrics

    def _compute_agg_metrics(self, logged_metrics: list[dict]) -> dict:
        num_distractors = np.array([x["num_distractors"] for x in logged_metrics])
        num_turns = np.array([x["num_turns"] for x in logged_metrics])

        agg_metrics = {
            # distractors
            "avg_num_distractors": np.mean(num_distractors),
            "stddev_num_distractors": np.std(num_distractors),
            "median_num_distractors": np.median(num_distractors),
            "max_num_distractors": np.max(num_distractors),
            "min_num_distractors": np.min(num_distractors),
            # turns
            "avg_num_turns": np.mean(num_turns),
            "stddev_num_turns": np.std(num_turns),
            "median_num_turns": np.median(num_turns),
            "max_num_turns": np.max(num_turns),
            "min_num_turns": np.min(num_turns),
            # task stats
            "false_positive_rate": np.mean([x["was_false_pos"] for x in logged_metrics]),
            "false_negative_rate": np.mean([x["was_false_neg"] for x in logged_metrics]),
            # distractor stats
            "avg_distractor_accuracy": np.nanmean(
                [x["distractor_accuracy"] for x in logged_metrics]
            ),
            # violation
            "violation_rate": np.mean([x["violation_occurred"] for x in logged_metrics]),
        }
        # necessary for serialization, json doesn't like np floats
        agg_metrics = {k: float(v) for k, v in agg_metrics.items()}
        return agg_metrics

    def _get_samples(self) -> list[dict]:
        samples = self.get_samples()
        samples = self.rng.sample(samples, min(self.n_samples, len(samples)))
        return samples
