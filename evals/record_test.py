import json
import tempfile

from evals.base import RunSpec
from evals.record import LocalRecorder


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


def test_http_recorder_final_report_no_events_falls_back_without_zerodivision() -> None:
    # Regression: HttpRecorder.record_final_report sends a report event that is
    # not appended to self._events. For a run that recorded no sample events,
    # self._events is empty, so a failed POST used to hit
    # `failed_requests / len(self._events)` -> ZeroDivisionError, which escaped
    # _send_event (record_final_report only catches RuntimeError) and bypassed
    # the intended LocalRecorder fallback.
    from unittest import mock

    from evals.record import HttpRecorder

    spec = RunSpec(
        completion_fns=[""],
        eval_name="",
        base_eval="",
        split="",
        run_config={},
        created_by="",
        run_id="test-run",
        created_at="",
    )
    fallback_path = tempfile.mktemp()
    recorder = HttpRecorder(
        url="http://localhost/does-not-exist",
        run_spec=spec,
        local_fallback_path=fallback_path,
        fail_percent_threshold=5,
    )
    # No sample events were recorded for this run.
    assert recorder._events == []

    with mock.patch("evals.record.requests.post", side_effect=Exception("connection refused")):
        # Must not raise ZeroDivisionError; must fall back to the LocalRecorder.
        recorder.record_final_report({"accuracy": 1.0})

    with open(fallback_path, "r", -1, "utf-8") as f:
        contents = f.read()
    assert "final_report" in contents
