"""
Implementation logic for Identifying Variables eval
"""
import logging
import random
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from evals.elsuite.identifying_variables import constants, graph_utils, prompts
from evals.elsuite.identifying_variables.metrics import (
    compute_fallout,
    compute_nDCG,
    compute_recall,
)
from evals.elsuite.identifying_variables.renderers import RENDERER_MAP
from evals.elsuite.identifying_variables.scripts.gen_data import gen_samples
from evals.elsuite.identifying_variables.structs import Answer, Sample
from evals.elsuite.identifying_variables.utils import json_to_sample, parse_solver_preds
from evals.eval import SolverEval
from evals.record import RecorderBase, record_metrics
from evals.solvers.solver import Solver, SolverResult
from evals.task_state import Message, TaskState

logging.getLogger("httpx").setLevel(logging.WARNING)


class IdentifyingVariables(SolverEval):
    def __init__(
        self,
        renderer: str,
        n_samples: Optional[int] = None,
        show_tree: bool = False,
        group_metrics: bool = False,
        debug: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.rng: random.Random = random.Random(self.seed)
        self.np_rng: np.random.Generator = np.random.default_rng(self.seed)
        self.renderer = RENDERER_MAP[renderer](rng=self.rng, np_rng=self.np_rng)
        self.renderer_variant = renderer
        self.n_samples = n_samples
        self.show_tree = show_tree
        self.task_description = self._build_task_description()
        self.group_metrics = group_metrics
        self.debug = debug

    def _build_task_description(self) -> str:
        decision_tree_section = ""
        if self.show_tree:
            decision_tree_section = prompts.DECISION_TREE_SECTION
        return prompts.TASK_DESCRIPTION.format(
            optional_decision_tree_section=decision_tree_section,
        ).strip()

    def eval_sample(self, solver: Solver, sample: Sample, rng: random.Random) -> None:
        message: Message = self._build_message(sample)

        task_state = TaskState(
            task_description=self.task_description,
            messages=[message],
            # to be used by the Random baseline solver only
            current_state={"variables": [var for var in sample.causal_graph.nodes]},
        )

        solver_result: SolverResult = solver(task_state)

        try:
            preds = parse_solver_preds(solver_result)
        except ValueError:  # in case of invalid solver output
            preds = None
        gold, num_not_ctrl = sample.gold_label, sample.num_not_ctrl

        metrics: Dict[str, float] = self._evaluate_sample(preds, gold, num_not_ctrl)

        record_metrics(
            **metrics,
            # hack: logviz doesn't support custom log fields, so logging as metric
            causal_graph=nx.to_dict_of_lists(sample.causal_graph),
            gold_answer=asdict(gold),
            n_hyps=sample.hypotheses.number_of_edges(),
            valid_hyp=gold.valid_hypothesis,
            num_not_ctrl=num_not_ctrl,
        )

    def run(self, recorder: RecorderBase) -> Dict[str, float]:
        samples: List[Dict] = self._get_samples()
        self.rng.shuffle(samples)
        self.eval_all_samples(recorder, samples)
        metrics: List[Dict] = recorder.get_metrics()

        return self._compute_agg_metrics(metrics)

    def _compute_agg_metrics(self, metrics: List[Dict]) -> Dict[str, float]:
        """
        Computes aggregate metrics across all samples
        """
        main_metrics = {
            "hyp_valid_acc": np.mean([x["hyp_valid_correct"] for x in metrics]),
            "violation_count": np.sum([x["violation"] for x in metrics]),
            "violation_rate": np.mean([x["violation"] for x in metrics]),
            # Some samples may be NaN for cases where the target hypothesis is invalid
            "ctrl_nDCG": np.nanmean([x["ctrl_nDCG"] for x in metrics]),
            "ctrl_recall": np.nanmean([x["ctrl_recall"] for x in metrics]),
            "ctrl_fallout": np.nanmean([x["ctrl_fallout"] for x in metrics]),
            "ind_acc": np.nanmean([x["ind_correct"] for x in metrics]),
            "dep_acc": np.nanmean([x["dep_correct"] for x in metrics]),
            "n_valid_hyp": np.sum([x["valid_hyp"] for x in metrics]),
        }
        if self.group_metrics:
            grouped_metrics = self._compute_grouped_metrics(metrics)
        else:
            grouped_metrics = {}

        total_metrics = {**main_metrics, **grouped_metrics}
        total_metrics = {k: float(v) for k, v in total_metrics.items()}
        return total_metrics

    def _compute_grouped_metrics(self, metrics: List[Dict]) -> Dict[str, float]:
        """
        Computes metrics aggregated across samples grouped by
          - number of variables
          - number of roots in random forest
          - number of control variables
          - number of hypotheses
          - max correlation depth
        """
        metric_to_agg_func = {
            "hyp_valid_acc": np.mean,
            "violation_count": np.sum,
            "violation_rate": np.mean,
            "ctrl_nDCG": np.nanmean,
            "ctrl_recall": np.nanmean,
            "ctrl_fallout": np.nanmean,
            "ind_acc": np.nanmean,
            "dep_acc": np.nanmean,
        }
        raw_metric_names = [
            "hyp_valid_correct",
            "violation",
            "violation",
            "ctrl_nDCG",
            "ctrl_recall",
            "ctrl_fallout",
            "ind_correct",
            "dep_correct",
        ]
        group_to_bins = {
            "n_vars": np.arange(constants.MIN_VARS, constants.MAX_VARS + 1),
            "n_roots": np.arange(1, constants.MAX_VARS + 1),
            "n_ctrl_vars": np.arange(0, (constants.MAX_VARS - 2) + 1),
            "n_hyps": np.arange(constants.MIN_HYPS, constants.MAX_HYPS + 1),
            "max_corr_depth": np.arange(1, constants.MAX_VARS),
        }
        grouped_metrics = {
            f"{metric}-{group}-{g_bin}": []
            for metric in metric_to_agg_func.keys()
            for group in group_to_bins.keys()
            for g_bin in group_to_bins[group]
        }
        for log_entry in metrics:
            causal_graph = nx.from_dict_of_lists(log_entry["causal_graph"], create_using=nx.DiGraph)
            ctrl_vars = log_entry["gold_answer"]["ctrl_vars"]
            dep_var = log_entry["gold_answer"]["dep_var"]
            group_to_bin = {
                "n_vars": causal_graph.number_of_nodes(),
                "n_roots": len(graph_utils.find_graph_roots(causal_graph)),
                "n_ctrl_vars": len(ctrl_vars) if ctrl_vars is not None else None,
                "n_hyps": log_entry["n_hyps"],
                "max_corr_depth": graph_utils.find_farthest_node(causal_graph, dep_var)[1]
                if dep_var is not None
                else None,
            }
            for group, g_bin in group_to_bin.items():
                if g_bin is not None:
                    for metric, raw_metric in zip(metric_to_agg_func.keys(), raw_metric_names):
                        grouped_metrics[f"{metric}-{group}-{g_bin}"].append(log_entry[raw_metric])

        # aggregate
        grouped_metrics = {
            k: metric_to_agg_func[k.split("-")[0]](v)
            # signal empty groups with np.nan
            if len(v) > 0 else np.nan
            for k, v in grouped_metrics.items()
        }
        return grouped_metrics

    def _evaluate_sample(self, preds: Optional[Answer], gold: Answer, num_not_ctrl: int) -> Dict:
        """
        If the gold hypothesis is invalid, then all other metrics are skipped, and we
        only evaluate whether the solver correctly identified the hypothesis as invalid.

        Mistakes are propagated: If the solver incorrectly identifies a hypothesis as
        invalid, then its missing answers for the remaining tasks are counted as wrong.

        In case of violations, the worst possible metrics are recorded, accounting for
        the gold hypothesis validity caveat above (e.g. if the gold hypothesis is
        invalid, then the worst case ctrl_nDCG is NaN since we'd skip this anyway,
        whereas if the gold hypothesis were valid, then the worst case ctrl_nDCG would
        be 0.0)
        """
        hyp_valid_correct = preds.valid_hypothesis == gold.valid_hypothesis if preds else False

        if gold.valid_hypothesis:
            ind_correct = preds.ind_var == gold.ind_var if preds else False
            dep_correct = preds.dep_var == gold.dep_var if preds else False
            ctrl_nDCG = (
                self._ctrl_vars_nDCG(preds.ctrl_vars, gold.ctrl_vars, num_not_ctrl)
                if preds and preds.ctrl_vars is not None
                else 0.0
            )
            ctrl_recall = (
                self._ctrl_vars_recall(preds.ctrl_vars, gold.ctrl_vars)
                if preds and preds.ctrl_vars is not None
                else 0.0
            )
            # not in final report, since experiments had already been run
            ctrl_fallout = (
                self._ctrl_vars_fallout(preds.ctrl_vars, gold.ctrl_vars, num_not_ctrl)
                if preds and preds.ctrl_vars is not None
                else 1.0
            )

        else:
            ctrl_nDCG = np.nan
            ctrl_recall = np.nan
            ctrl_fallout = np.nan
            ind_correct = np.nan
            dep_correct = np.nan

        return {
            "ctrl_nDCG": ctrl_nDCG,
            "ctrl_recall": ctrl_recall,
            "ctrl_fallout": ctrl_fallout,
            "ind_correct": ind_correct,
            "dep_correct": dep_correct,
            "hyp_valid_correct": hyp_valid_correct,
            "violation": preds is None,
        }

    def _ctrl_vars_fallout(self, preds: List[str], gold: List[str], num_not_ctrl: int) -> float:
        return compute_fallout(set(preds), set(gold), num_not_ctrl)

    def _ctrl_vars_recall(self, preds: List[str], gold: List[str]) -> float:
        return compute_recall(set(preds), set(gold))

    def _ctrl_vars_nDCG(self, preds: List[str], gold: List[str], num_not_ctrl: int) -> float:
        best = [1.0] * len(gold)
        ranking = [1.0 if var in gold else -1.0 for var in preds]
        worst_case_ctrl = [-1.0] * num_not_ctrl
        return compute_nDCG(ranking, best, worst_case_ctrl)

    def _build_message(self, sample: Sample) -> Message:
        observations: str = self.renderer.render_obs(sample)
        hypotheses: List[str] = self._render_hypotheses(sample.hypotheses)
        target_hypothesis: str = self._render_hypothesis(sample.target_hypothesis)

        message_content = prompts.SAMPLE_MESSAGE.format(
            observations=observations,
            hypotheses=hypotheses,
            target_hypothesis=target_hypothesis,
        ).strip()
        message = Message("user", content=message_content)

        return message

    def _render_hypotheses(self, hypotheses: nx.DiGraph) -> List[str]:
        hyp_list = [(n, adj) for n in hypotheses for adj in hypotheses[n]]
        return [self._render_hypothesis(h) for h in hyp_list]

    def _render_hypothesis(self, hypothesis: Tuple[str, str]) -> str:
        hyp_template = self.rng.choice(prompts.hypothesis_templates)
        rendered_hyp = hyp_template.format(ind=hypothesis[0], dep=hypothesis[1])
        return rendered_hyp

    def _get_samples(self) -> List[Sample]:
        if self.debug:
            return gen_samples(n_samples=1000, signal_noise_ratio=None, np_rng=self.np_rng)

        dict_samples = self.get_samples()
        if self.n_samples is not None:
            assert (
                len(dict_samples) >= self.n_samples
            ), f"Can't get {self.n_samples} samples from a dataset with {len(dict_samples)} samples"
            np.random.default_rng(seed=self.seed).shuffle(dict_samples)
            dict_samples = dict_samples[: self.n_samples]
        samples = [json_to_sample(dict_sample) for dict_sample in dict_samples]
        return samples
