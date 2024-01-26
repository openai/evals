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
