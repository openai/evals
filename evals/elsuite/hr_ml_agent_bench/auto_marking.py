import importlib
import json
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path


@dataclass(frozen=True)
class EvaluationResult:
    # Raw scores in the original unit of the task.
    model_score: float
    naive_baseline_score: float
    human_baseline_score: float
    # Normalized scores are in the range [0, 1] where higher is better.
    model_score_normalized: float
    naive_baseline_score_normalized: float
    human_baseline_score_normalized: float
    # Human-relative scores are in the range [0, 1] where 0 is the naive
    # baseline and 1 is the human baseline.
    model_score_humanrelative: float


def grade_submission(log_dir: Path, task_name: str) -> EvaluationResult:
    """
    Grades the submission in `log_dir` using a task-specific grading script.
    """

    # TODO: refactor this to not dynamically import the grade module
    grading_module = importlib.import_module(
        f"evals.elsuite.hr_ml_agent_bench.benchmarks.{task_name}.scripts.grade"
    )

    # Get baselines
    naive_baseline_score_raw = grading_module.get_naive_baseline_score()
    naive_baseline_score_normalized = grading_module.normalize_score(naive_baseline_score_raw)
    human_baseline_score_raw = grading_module.get_human_baseline_score()
    human_baseline_score_normalized = grading_module.normalize_score(human_baseline_score_raw)

    traces = list(log_dir.glob("**/trace.json"))

    assert len(traces) == 1, f"Expected to find exactly one submission. Found {len(traces)}."

    best_raw_score = naive_baseline_score_raw
    best_normalized_score = naive_baseline_score_normalized

    for trace in traces:
        with open(trace) as f:
            contents = f.read()

        try:
            data = json.loads(contents)
        except JSONDecodeError:
            continue

        n_steps = len(data["steps"])

        for step in range(n_steps):
            submission_dir = trace.parent / "traces" / f"step_{step}_files"
            raw_score = grading_module.get_score(submission_dir)
            normalized_score = grading_module.normalize_score(raw_score)

            if normalized_score > best_normalized_score:
                best_raw_score = raw_score
                best_normalized_score = normalized_score

    # Calculate final human-relative score using normalized scores
    model_score_humanrelative = (best_normalized_score - naive_baseline_score_normalized) / (
        human_baseline_score_normalized - naive_baseline_score_normalized
    )

    return EvaluationResult(
        model_score=best_raw_score,
        naive_baseline_score=naive_baseline_score_raw,
        human_baseline_score=human_baseline_score_raw,
        model_score_normalized=best_normalized_score,
        naive_baseline_score_normalized=naive_baseline_score_normalized,
        human_baseline_score_normalized=human_baseline_score_normalized,
        model_score_humanrelative=model_score_humanrelative,
    )
