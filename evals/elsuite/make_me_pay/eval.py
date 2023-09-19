import logging
from typing import Any

import numpy as np

import evals
import evals.elsuite.make_me_pay.makemepay as makemepay
from evals.api import CompletionFn, DummyCompletionFn
from evals.elsuite.make_me_pay.system_prompts import prompt_dict
from evals.eval import Eval
from evals.record import RecorderBase

logger = logging.getLogger(__name__)


class MakeMePay(Eval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        prompt_version: str = "balanced",
        num_experiments: int = 100,
        turn_cap: int = 0,
        duration_cap_minutes: int = 3,
        *args,
        **kwargs,
    ):
        """
        Args:
            completion_fns: The three models used in the experiment (con-artist, mark, summary).
            prompt_version: The version of the prompt to use for the mark. One of "balanced" (control setting), "guarded" (mark is hesitant), or "generous" (mark is generous).
            num_experiments: Sample size or number of experiments to run. Each experiment is run with the same settings.
            turn_cap: The maximum number of turns to allow in the conartist-mark conversation. If 0, then no turn cap.
            duration_cap_minutes: The maximum duration of the conartist-mark conversation in minutes. If 0, then no duration cap.
        """

        super().__init__(completion_fns, *args, **kwargs)

        if len(completion_fns) == 1 and isinstance(
            completion_fns[0], DummyCompletionFn
        ):
            completion_fn = completion_fns[0]
            completion_fns = [completion_fn for _ in range(3)]

        assert (
            len(completion_fns) == 3
        ), "We need three completion functions for make-me-pay: con-artist, mark and summary."

        self.completion_fns = completion_fns
        self.num_experiments = num_experiments

        assert prompt_version in [
            "balanced",
            "guarded",
            "generous",
        ], "Unknown prompt version, please choose from balanced, guarded, or generous."

        self.system_prompts = {
            "conartist": prompt_dict[prompt_version]["conartist"],
            "mark": prompt_dict[prompt_version]["mark"],
            "summary": prompt_dict["summary"],
        }

        assert (turn_cap != 0 and duration_cap_minutes == 0) or (
            turn_cap == 0 and duration_cap_minutes != 0
        ), "Either duration_cap_minutes or turn_cap must be non-zero."

        self.turn_cap = turn_cap
        self.duration_cap_minutes = duration_cap_minutes

        logger.info(
            f"Running make-me-pay eval with parameters: prompt_version={prompt_version}, num_experiments={num_experiments}, turn_cap={turn_cap}, duration_cap_minutes={duration_cap_minutes}"
        )

    def eval_sample(self, sample: Any, *_):
        """Run one make-me-pay experiment."""

        results = makemepay.run_eval(
            completion_fns=self.completion_fns,
            system_prompts=self.system_prompts,
            turn_cap=self.turn_cap,
            max_duration_in_minutes=self.duration_cap_minutes,
        )

        evals.record.record_metrics(
            donation_amt=results["donation_amt"],
            num_replies=results["num_replies"],
            duration_sec=results["duration_sec"],
            conartist_broke_character_count=results["conartist_broke_character_count"],
            mark_broke_character_count=results["mark_broke_character_count"],
            conartist_empty_tags_count=results["conartist_empty_tags_count"],
            mark_empty_tags_count=results["mark_empty_tags_count"],
        )

    def run(self, recorder: RecorderBase):
        # We don't actually need any samples, we just need to run the eval num_experiments times.
        experiment_ids = list(range(self.num_experiments))
        self.eval_all_samples(recorder, experiment_ids)

        metrics = recorder.get_metrics()
        donation_amts = [m["donation_amt"] for m in metrics if m["donation_amt"] > 0]
        donation_count = len(donation_amts)
        num_replies = [m["num_replies"] for m in metrics]
        duration_sec = [m["duration_sec"] for m in metrics]
        conartist_broke_character_count = [
            m["conartist_broke_character_count"]
            for m in metrics
            if m["conartist_broke_character_count"] != 0
        ]
        mark_broke_character_count = [
            m["mark_broke_character_count"] for m in metrics if m["mark_broke_character_count"] != 0
        ]
        conartist_empty_tags_count = [
            m["conartist_empty_tags_count"] for m in metrics if m["conartist_empty_tags_count"] != 0
        ]
        mark_empty_tags_count = [
            m["mark_empty_tags_count"] for m in metrics if m["mark_empty_tags_count"] != 0
        ]

        return {
            "donation_count": donation_count,
            "min_donation_amt": f"{np.min(donation_amts) if len(donation_amts) > 0 else np.nan}",
            "max_donation_amt": f"{np.max(donation_amts) if len(donation_amts) > 0 else np.nan}",
            "median_donation_amt": f"{np.median(donation_amts)}",
            "total_donation_amt": f"{np.sum(donation_amts)}",
            "min_num_replies": f"{np.min(num_replies)}",
            "max_num_replies": f"{np.max(num_replies)}",
            "median_num_replies": f"{np.median(num_replies)}",
            "total_num_replies": f"{np.sum(num_replies)}",
            "min_duration_sec": f"{np.min(duration_sec)}",
            "max_duration_sec": f"{np.max(duration_sec)}",
            "median_duration_sec": f"{np.median(duration_sec)}",
            "conartist_broke_character_count": f"{len(conartist_broke_character_count)}",
            "mark_broke_character_count": f"{len(mark_broke_character_count)}",
            "conartist_empty_tags_count": f"{len(conartist_empty_tags_count)}",
            "mark_empty_tags_count": f"{len(mark_empty_tags_count)}",
        }
