from pathlib import Path

from evals.elsuite.lambada import Lambada
from evals.record import DummyRecorder
from evals.utils.test import TestCompletionFn


def _score(completion: str) -> bool:
    evaluator = Lambada(
        completion_fns=[TestCompletionFn(completion)],
        subset="en",
        eval_registry_path=Path("."),
    )
    recorder = DummyRecorder(None)
    with recorder.as_default_recorder("sample"):
        evaluator.eval_sample({"text": "The cat"}, None)
    return recorder.get_events("match")[0].data["correct"]


def test_lambada_accepts_exact_word_and_continuation() -> None:
    assert _score("cat")
    assert _score("cat.")
    assert _score("cat is sleeping")


def test_lambada_rejects_longer_words_with_same_prefix() -> None:
    assert not _score("caterpillar")
    assert not _score("cat2")
