import evals
from evals.base import RunSpec
from evals.record import RecorderBase


def make_run_spec() -> RunSpec:
    return RunSpec(
        completion_fns=["dummy"],
        eval_name="example.dev.v0",
        base_eval="example",
        split="dev",
        run_config={},
        created_by="test",
    )


def test_current_run_id_is_none_without_active_recorder() -> None:
    assert evals.current_run_id() is None


def test_current_run_id_exposes_active_recorder_run_id() -> None:
    run_spec = make_run_spec()
    recorder = RecorderBase(run_spec)

    with recorder.as_default_recorder("example.dev.0"):
        assert evals.current_run_id() == run_spec.run_id

    assert evals.current_run_id() is None


def test_current_run_id_tracks_nested_recorder_contexts() -> None:
    outer_spec = make_run_spec()
    inner_spec = make_run_spec()
    outer = RecorderBase(outer_spec)
    inner = RecorderBase(inner_spec)

    with outer.as_default_recorder("example.dev.0"):
        assert evals.current_run_id() == outer_spec.run_id
        with inner.as_default_recorder("example.dev.1"):
            assert evals.current_run_id() == inner_spec.run_id
        assert evals.current_run_id() == outer_spec.run_id
