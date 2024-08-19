import logging
import random
import re
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import scipy

import evals
from evals.api import CompletionFn
from evals.elsuite.function_deduction import prompts
from evals.eval import SolverEval
from evals.solvers.solver import Solver
from evals.task_state import Message, TaskState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Sample:
    sample_ix: int
    code: str
    complexity: int
    range: List[int]
    values: List[int]


@dataclass
class CurrentState:
    """This class tracks all the information from the dialog with the model.

    Some things are tracked to make writing solvers easier.
    Other are tracked for metrics.
    """

    n_rounds: int
    mode: str
    test_inputs: tuple[int, int, int]
    success: bool = False
    known_values: dict[int, int] = field(default_factory=dict)
    negative_known_values: dict[int, int] = field(default_factory=dict)
    ask_rounds: int = 0
    guess_rounds: int = 0
    incorrect_format_rounds: int = 0
    parsed_responses: list[tuple[int]] = field(default_factory=list)

    @property
    def round_ix(self):
        return self.ask_rounds + self.guess_rounds + self.incorrect_format_rounds

    def ask_update(self, input_: int, value: Optional[int]) -> None:
        self.ask_rounds += 1
        self.parsed_responses.append((input_,))
        if value is not None:
            self.known_values[input_] = value

    def guess_update(
        self, guessed_ints: tuple[int, int, int], expected_ints: tuple[int, int, int]
    ) -> None:
        self.guess_rounds += 1
        self.parsed_responses.append(guessed_ints)
        if guessed_ints == expected_ints:
            self.success = True

        if self.mode == "easy":
            for test, guess, correct in zip(self.test_inputs, guessed_ints, expected_ints):
                if guess == correct:
                    self.known_values[test] = guess
                else:
                    self.negative_known_values[test] = guess


class FunctionDeductionEval(SolverEval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        mode: Literal["easy", "hard"],
        n_rounds: int,
        n_samples: Optional[int] = None,
        n_repeat: int = 3,
        failed_sample_rounds: Optional[int] = None,
        seed: Optional[int] = None,
        samples_jsonl: str = "function_deduction/data.jsonl",
        *args,
        **kwargs,
    ):
        super().__init__(completion_fns, seed=seed, samples_jsonl=samples_jsonl, *args, **kwargs)

        self.mode = mode
        self.n_rounds = n_rounds
        self.n_samples = n_samples
        self.n_repeat = n_repeat

        #   This is used for the main metric - "how many rounds for a sample that was not solved?"
        self.failed_sample_rounds = (
            failed_sample_rounds if failed_sample_rounds is not None else n_rounds * 2
        )

    def eval_sample(self, solver: Solver, sample: Sample, rng: random.Random):
        test_inputs = rng.sample(range(101), 3)
        values = sample.values
        expected = tuple(sample.values[test_input] for test_input in test_inputs)

        cs = CurrentState(self.n_rounds, self.mode, test_inputs)
        task_state = TaskState(
            prompts.task_description.format(inputs=test_inputs, n_rounds=self.n_rounds),
            current_state=cs,
        )

        for round_ix in range(self.n_rounds):
            raw_response = solver(task_state).output
            try:
                ints = self._parse_raw_response(raw_response)
            except ValueError:
                cs.incorrect_format_rounds += 1
                answer = prompts.incorrect_format
            else:
                if len(ints) == 1:
                    ask = ints[0]
                    result = values[ask] if ask not in test_inputs else None
                    cs.ask_update(ask, result)
                    if result is None:
                        answer = prompts.test_input_not_allowed.format(inputs=test_inputs)
                    else:
                        answer = prompts.new_value.format(in_=ask, out=result)
                else:
                    cs.guess_update(ints, expected)
                    if cs.success:
                        break
                    else:
                        answer = self._bad_guess_answer(test_inputs, ints, expected)

            task_state.messages += [
                Message("assistant", raw_response),
                Message("system", answer),
            ]

        evals.record.record_metrics(
            sample_ix=sample.sample_ix,
            success=cs.success,
            num_rounds=cs.round_ix if cs.success else None,
            ask_rounds=cs.ask_rounds,
            guess_rounds=cs.guess_rounds,
            incorrect_format_rounds=cs.incorrect_format_rounds,
            repeated_rounds=len(cs.parsed_responses) - len(set(cs.parsed_responses)),
            code="lambda x: " + sample.code,
            complexity=sample.complexity,
        )

    def run(self, recorder: evals.record.Recorder):
        samples = self.get_samples()

        #   Add copies according to self.n_repeat
        #   NOTE: we have copies next to each other -> more convenient when reading in logviz
        copied_samples = [sample for sample in samples for _ in range(self.n_repeat)]
        logger.info(
            f"{len(samples)} unique samples, {self.n_repeat} attempts for each sample, {len(copied_samples)} total samples"
        )
        self.eval_all_samples(recorder, copied_samples)
        metrics = recorder.get_metrics()

        adjusted_rounds = [x["num_rounds"] or self.failed_sample_rounds for x in metrics]
        main_metric = sum(adjusted_rounds) / len(metrics)
        result = {
            "adjusted_avg_score": main_metric,
            "sem_adjusted_avg_score": self._calculate_sem(adjusted_rounds),
        }

        result.update(self._get_success_metrics(metrics))
        result.update(self._get_sample_std(metrics))
        for name in ("ask_rounds", "guess_rounds", "incorrect_format_rounds"):
            result[f"avg_{name}"] = sum(x[name] for x in metrics) / len(metrics)
            result[f"sem_avg_{name}"] = self._calculate_sem([x[name] for x in metrics])
        result.update(self._get_complexity_tests(metrics))
        result.update(self._get_per_complexity_metrics(metrics))

        return result

    def _calculate_sem(self, values: list) -> float:
        return np.std(values) / np.sqrt(len(values))

    def _get_success_metrics(self, metrics):
        success = [x for x in metrics if x["success"]]
        return {
            "solved_ratio": round(len(success) / len(metrics), 2),
            "sem_solved_ratio": self._calculate_sem([x["success"] for x in metrics]),
            "solved": len(success),
            "samples": len(metrics),
            "avg_success_rounds": round(sum(x["num_rounds"] for x in success) / len(success), 2)
            if success
            else None,
            "sem_avg_success_rounds": self._calculate_sem([x["num_rounds"] for x in success])
            if success
            else None,
        }

    def _get_sample_std(self, metrics):
        adjusted = []
        no_failed = []
        solved_ratio_if_any_solved = []
        sample_ixs = set(metric["sample_ix"] for metric in metrics)
        for sample_ix in sample_ixs:
            sample_metrics = [metric for metric in metrics if metric["sample_ix"] == sample_ix]
            sample_adjusted = [
                metric["num_rounds"] or self.failed_sample_rounds for metric in sample_metrics
            ]
            sample_no_failed = [
                metric["num_rounds"] for metric in sample_metrics if metric["success"]
            ]
            solved_ratio = sum(1 for metric in sample_metrics if metric["success"]) / len(
                sample_metrics
            )

            if len(sample_adjusted) > 1:
                adjusted.append(np.std(sample_adjusted))
            if len(sample_no_failed) > 1:
                no_failed.append(np.std(sample_no_failed))
            if solved_ratio:
                solved_ratio_if_any_solved.append(solved_ratio)

        return {
            "avg_sample_rounds_std_adjusted": sum(adjusted) / len(adjusted) if adjusted else None,
            "avg_sample_rounds_std_no_failed": sum(no_failed) / len(no_failed)
            if no_failed
            else None,
            #   This is just solved_ratio but excluding samples that had no succesful attempt.
            #   So 1 is full stability (i.e. if sample was solved once, it will be solved always),
            #   and (1/self.n_repeat) is "no sample was solved more than once"
            "solved_ratio_if_any_solved": sum(solved_ratio_if_any_solved)
            / len(solved_ratio_if_any_solved)
            if solved_ratio_if_any_solved
            else None,
        }

    def _get_complexity_tests(self, metrics):
        solved = [x["complexity"] for x in metrics if x["success"]]
        not_solved = [x["complexity"] for x in metrics if not x["success"]]
        result = {
            "solved_avg_complexity": sum(solved) / len(solved) if solved else None,
            "not_solved_avg_complexity": sum(not_solved) / len(not_solved) if not_solved else None,
        }

        #   This tests if solved have lower complexity than non-solved
        if solved and not_solved:
            _, p_value = scipy.stats.mannwhitneyu(solved, not_solved, alternative="less")
        else:
            p_value = None
        result["solved_or_not_mann_whitney_u_p_value"] = p_value

        #   TODO: add more complexity-related metrics, such as correlation or linear regression coefficient.
        #         Leaving this for the future because we might want to change how the complexity is calculated,
        #         or generally improve the concept somehow.

        return result

    def _get_per_complexity_metrics(self, all_metrics):
        complexity_values = sorted(x["complexity"] for x in all_metrics)
        result = {}
        for complexity in complexity_values:
            metrics = [x for x in all_metrics if x["complexity"] == complexity]
            result[f"complexity_{complexity}"] = self._get_success_metrics(metrics)
        return result

    def _parse_raw_response(self, response: str) -> Union[Tuple[int], Tuple[int, int, int]]:
        #   Remove all non-numbers first. This way we accept also e.g. "1, 2, 3", "[1, 2, 3]", '"1", "2", "3"' etc.
        response = re.sub(r"[^0-9\s-]", "", response)

        vals = tuple(int(x) for x in response.split())
        if len(vals) not in (1, 3):
            raise ValueError("Expected 1 or 3 integers")
        if len(vals) == 1 and not 0 <= vals[0] <= 100:
            raise ValueError("Single int should be between 0 and 100")
        return vals

    def _bad_guess_answer(self, test_inputs, guessed, expected) -> str:
        correct = [test_inputs[i] for i in range(0, 3) if guessed[i] == expected[i]]
        incorrect = [x for x in test_inputs if x not in correct]
        assert incorrect, "This is not a bad answer"

        if self.mode == "hard":
            return "This is not the correct answer. At least one of the values is wrong."
        elif self.mode == "easy":
            if len(correct) == 0:
                return "All numbers are wrong."
            elif len(correct) == 1:
                return f"Your guess is correct for {correct[0]} and incorrect for {incorrect[0]} and {incorrect[1]}"
            elif len(correct) == 2:
                return f"Your guess is correct for {correct[0]} and {correct[1]} and incorrect for {incorrect[0]}"
        else:
            raise ValueError(f"Unknown mode {self.mode}")

    def get_samples(self) -> List[Sample]:
        samples = super().get_samples()

        if self.n_samples is not None:
            assert (
                len(samples) >= self.n_samples
            ), f"Can't get {self.n_samples} samples from a dataset with {len(samples)} samples"
            np.random.default_rng(seed=self.seed).shuffle(samples)
            samples = samples[: self.n_samples]
        return [Sample(**sample_dict) for sample_dict in samples]
