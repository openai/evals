from mock import patch
from evals.api import DummyCompletionFn
from evals.elsuite.basic.fuzzy_match import FuzzyMatch
from evals.record import DummyRecorder


def test_eval_sample_str():
    eval = FuzzyMatch(
        completion_fns=[DummyCompletionFn()],
        samples_jsonl="",
    )

    recorder = DummyRecorder(None)
    with recorder.as_default_recorder("x"), patch.object(
        recorder, "record_metrics", wraps=recorder.record_metrics
    ) as record_metrics:
        eval.eval_sample(dict(input="Hello", ideal="world"), None)
        record_metrics.assert_called_once_with(accuracy=0.0, f1_score=0.0)


def test_eval_sample_list():
    eval = FuzzyMatch(
        completion_fns=[DummyCompletionFn()],
        samples_jsonl="",
    )

    recorder = DummyRecorder(None)
    with recorder.as_default_recorder("x"), patch.object(
        recorder, "record_metrics", wraps=recorder.record_metrics
    ) as record_metrics:
        eval.eval_sample(dict(input="Hello", ideal=["world", "dummy"]), None)
        record_metrics.assert_called_once_with(accuracy=1.0, f1_score=0.4)
