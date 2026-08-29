import json
import tempfile
from typing import List
from unittest.mock import patch

from evals.base import RunSpec
from evals.record import HttpRecorder, LocalRecorder


def test_passes_hidden_data_field_to_jsondumps() -> None:
    tmp_file = tempfile.mktemp()
    spec = RunSpec(
        completion_fns=[""],
        eval_name="",
        base_eval="",
        split="",
        run_config={},
        created_by="",
        run_id="",
        created_at="",
    )
    local_recorder = LocalRecorder(tmp_file, spec, ["should_be_hidden"])
    local_recorder.record_event(
        "raw_sample", {"should_be_hidden": 1, "should_not_be_hidden": 2}, sample_id="test"
    )
    local_recorder.flush_events()
    with open(tmp_file, "r", -1, "utf-8") as f:
        first_line = f.readline()
        assert len(first_line) > 0
        second_line = json.loads(f.readline())
        assert second_line["data"] == {"should_not_be_hidden": 2}


def _make_run_spec() -> RunSpec:
    return RunSpec(
        completion_fns=[""],
        eval_name="",
        base_eval="",
        split="",
        run_config={},
        created_by="",
        run_id="test-run",
        created_at="",
    )


def _read_fallback_events(path: str) -> List[dict]:
    """Read every event line from a LocalRecorder-style log, skipping the spec header."""
    rows = []
    with open(path, "r", -1, "utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "spec" in row:
                continue
            rows.append(row)
    return rows


class _FakeResponse:
    """Minimal stand-in for requests.Response; HttpRecorder only reads .ok and .text."""

    def __init__(self, ok: bool, text: str = ""):
        self.ok = ok
        self.text = text


def test_http_recorder_falls_back_on_non_ok_response() -> None:
    """
    Regression test for #1819: a non-OK HTTP response (e.g. a 500) must be
    treated as a delivery failure, just like a connection error. The batch
    must be written to the local fallback file, not silently discarded.
    """
    fallback_path = tempfile.mktemp()
    run_spec = _make_run_spec()

    with patch("evals.record.requests.post", return_value=_FakeResponse(ok=False, text="boom")):
        recorder = HttpRecorder(
            url="http://example.invalid/ingest",
            run_spec=run_spec,
            local_fallback_path=fallback_path,
            fail_percent_threshold=100,  # never raise; we're only checking persistence here
            batch_size=10,
        )
        with recorder.as_default_recorder("sample-1"):
            for _ in range(5):
                recorder.record_match(True, expected="a", picked="a")
        recorder.flush_events()

    fallback_events = _read_fallback_events(fallback_path)
    assert len(fallback_events) == 5
    assert recorder.failed_requests == 1
    assert recorder.total_requests == 1


def test_http_recorder_success_does_not_use_fallback() -> None:
    fallback_path = tempfile.mktemp()
    run_spec = _make_run_spec()

    with patch("evals.record.requests.post", return_value=_FakeResponse(ok=True)):
        recorder = HttpRecorder(
            url="http://example.invalid/ingest",
            run_spec=run_spec,
            local_fallback_path=fallback_path,
            fail_percent_threshold=5,
            batch_size=10,
        )
        with recorder.as_default_recorder("sample-1"):
            for _ in range(3):
                recorder.record_match(True, expected="a", picked="a")
        recorder.flush_events()

    assert _read_fallback_events(fallback_path) == []
    assert recorder.failed_requests == 0


def test_http_recorder_raises_after_persisting_when_threshold_exceeded() -> None:
    """
    The threshold breach must still raise (fail loud), but only *after* the
    triggering batch has already been saved locally -- the raise must never
    be the reason data goes unsaved.
    """
    fallback_path = tempfile.mktemp()
    run_spec = _make_run_spec()

    with patch("evals.record.requests.post", return_value=_FakeResponse(ok=False, text="boom")):
        recorder = HttpRecorder(
            url="http://example.invalid/ingest",
            run_spec=run_spec,
            local_fallback_path=fallback_path,
            fail_percent_threshold=0,  # any failure trips it
            batch_size=10,
        )
        with recorder.as_default_recorder("sample-1"):
            recorder.record_match(True, expected="a", picked="a")

        raised = False
        try:
            recorder.flush_events()
        except RuntimeError:
            raised = True
        assert raised, "expected flush_events() to raise once the failure threshold was exceeded"

    assert len(_read_fallback_events(fallback_path)) == 1


def test_http_recorder_does_not_lose_earlier_failed_batch() -> None:
    """
    Regression test for #1819: a batch that fails but does not, by itself,
    push the failure rate over the threshold must still be saved locally --
    it must not be silently dropped just because a *later* batch is the one
    that finally trips the threshold.
    """
    fallback_path = tempfile.mktemp()
    run_spec = _make_run_spec()

    responses = [
        _FakeResponse(ok=True),  # batch 1: succeeds
        _FakeResponse(ok=False, text="boom"),  # batch 2: fails; 1/2 = 50% does not exceed 50%
        _FakeResponse(ok=False, text="boom"),  # batch 3: fails; 2/3 = 66.7% exceeds 50%
    ]

    with patch("evals.record.requests.post", side_effect=responses):
        recorder = HttpRecorder(
            url="http://example.invalid/ingest",
            run_spec=run_spec,
            local_fallback_path=fallback_path,
            fail_percent_threshold=50,
            batch_size=2,
        )
        with recorder.as_default_recorder("sample-1"):
            for _ in range(6):
                recorder.record_match(True, expected="a", picked="a")

        raised = False
        try:
            recorder.flush_events()
        except RuntimeError:
            raised = True
        assert raised

    # Batch 2 (2 events) and batch 3 (2 events) both failed and must both be
    # saved -- previously, only the batch that tripped the threshold was
    # saved, silently losing batch 2's events.
    assert len(_read_fallback_events(fallback_path)) == 4


def test_http_recorder_final_report_falls_back_on_failure() -> None:
    fallback_path = tempfile.mktemp()
    run_spec = _make_run_spec()

    with patch("evals.record.requests.post", return_value=_FakeResponse(ok=False, text="boom")):
        recorder = HttpRecorder(
            url="http://example.invalid/ingest",
            run_spec=run_spec,
            local_fallback_path=fallback_path,
            fail_percent_threshold=100,
            batch_size=10,
        )
        recorder.record_final_report({"accuracy": 0.5})

    fallback_events = _read_fallback_events(fallback_path)
    assert any(row.get("final_report") == {"accuracy": 0.5} for row in fallback_events)