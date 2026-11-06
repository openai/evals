from contextvars import Context

from evals.base import RunSpec
from evals.record import RecorderBase, current_sample_id, default_recorder


def make_recorder() -> RecorderBase:
    return RecorderBase(
        RunSpec(
            completion_fns=["dummy"],
            eval_name="test.dev",
            base_eval="test",
            split="dev",
            run_config={},
            created_by="test",
        )
    )


def test_current_sample_id_returns_active_value() -> None:
    def check() -> None:
        recorder = make_recorder()
        with recorder.as_default_recorder("sample-1"):
            assert current_sample_id() == "sample-1"

    Context().run(check)


def test_nested_recorder_context_restores_outer_context() -> None:
    def check() -> None:
        outer = make_recorder()
        inner = make_recorder()

        with outer.as_default_recorder("outer"):
            assert default_recorder() is outer
            assert current_sample_id() == "outer"

            with inner.as_default_recorder("inner"):
                assert default_recorder() is inner
                assert current_sample_id() == "inner"

            assert default_recorder() is outer
            assert current_sample_id() == "outer"

        assert default_recorder() is None
        assert current_sample_id() is None

    Context().run(check)


def test_recorder_context_restores_after_exception() -> None:
    def check() -> None:
        outer = make_recorder()
        inner = make_recorder()

        with outer.as_default_recorder("outer"):
            try:
                with inner.as_default_recorder("inner"):
                    raise RuntimeError("boom")
            except RuntimeError:
                pass

            assert default_recorder() is outer
            assert current_sample_id() == "outer"

        assert default_recorder() is None
        assert current_sample_id() is None

    Context().run(check)
