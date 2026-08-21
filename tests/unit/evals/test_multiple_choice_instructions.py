import random
from pathlib import Path

from evals.elsuite.multiple_choice import MultipleChoice, Sample
from evals.record import DummyRecorder
from evals.utils.test import TestCompletionFn


def test_multiple_choice_accepts_none_instructions() -> None:
    evaluator = MultipleChoice(
        completion_fns=[TestCompletionFn("A")],
        dataset="unused",
        instructions=None,
        eval_registry_path=Path("."),
    )
    recorder = DummyRecorder(None)

    with recorder.as_default_recorder("sample"):
        evaluator.eval_sample(
            Sample(question="Question?", answers=["correct", "wrong"], label=0),
            random.Random(0),
        )

    assert evaluator.instructions == ""
    assert len(recorder.get_events("match")) == 1
