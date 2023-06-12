"""
This file defines the recorder classes which log eval results in different ways,
such as to a local JSON file or to a remote Snowflake database.

If you would like to implement a custom recorder, you can see how the
`LocalRecorder` and `Recorder` classes inherit from the `RecorderBase` class and
override certain methods.
"""
import atexit
import contextlib
import dataclasses
import logging
import threading
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, List, Optional, Sequence

import blobfile as bf

import evals
from evals.base import RunSpec
from evals.data import jsondumps
from evals.utils.misc import t
from evals.utils.snowflake import SnowflakeConnection

logger = logging.getLogger(__name__)

MIN_FLUSH_EVENTS = 100
MAX_SNOWFLAKE_BYTES = 16 * 10**6
MIN_FLUSH_SECONDS = 10

_default_recorder: ContextVar[Optional["RecorderBase"]] = ContextVar(
    "default_recorder", default=None
)


def default_recorder() -> Optional["RecorderBase"]:
    return _default_recorder.get()


@dataclasses.dataclass
class Event:
    run_id: str
    event_id: int
    sample_id: Optional[str]
    type: str
    data: dict
    created_by: str
    created_at: str


class RecorderBase:
    """
    The standard events for which recording methods are provided are:
    - `match`: A match or non match, as specified by the `correct` bool, between
        the `expected` and `picked` results.
    - `embedding`: An embedding of the `prompt` of type `embedding_type`.
    - `sampling`: What was `sampled` from the model given the input `prompt`.
    - `cond_logp`: The conditional log probability, as `logp`, of the
        `completion` from the model given the input `prompt`.
    - `pick_option`: The option `picked` by the model out of the valid `options`
        given the input `prompt`.
    - `raw`: A raw sample specified by the `data`.
    - `metrics`: A set of metrics specified by the `kwargs`.
    - `error`: An `error` along with an accompanying `msg`.
    - `extra`: Any extra `data` of interest to be recorded.
    For these events, helper methods are defined at the bottom of this file.
    More generally, you can record any event by calling `record_event` with the
    event `type` and `data`.
    Finally, you can also record a final report using `record_final_report`.
    """

    def __init__(
        self,
        run_spec: evals.base.RunSpec,
    ) -> None:
        self._sample_id: ContextVar[Optional[int]] = ContextVar("_sample_id", default=None)
        self.run_spec = run_spec
        self._events: List[Event] = []
        self._last_flush_time = time.time()
        self._flushes_done = 0
        self._written_events = 0
        self._flushes_started = 0
        self._event_lock = threading.Lock()
        self._paused_ids: List[str] = []
        atexit.register(self.flush_events)

    @contextlib.contextmanager
    def as_default_recorder(self, sample_id: str):
        sample_id_token = self._sample_id.set(sample_id)
        default_recorder_token = _default_recorder.set(self)
        yield
        _default_recorder.reset(default_recorder_token)
        self._sample_id.reset(sample_id_token)

    def current_sample_id(self) -> Optional[str]:
        return self._sample_id.get()

    def pause(self):
        sample_id = self.current_sample_id()
        with self._event_lock:
            if sample_id not in self._paused_ids:
                self._paused_ids.append(sample_id)

    def unpause(self):
        sample_id = self.current_sample_id()
        with self._event_lock:
            if sample_id in self._paused_ids:
                self._paused_ids.remove(sample_id)

    def is_paused(self, sample_id: str = None):
        if sample_id is None:
            sample_id = self.current_sample_id()
        with self._event_lock:
            return sample_id in self._paused_ids

    def get_events(self, type: str) -> Sequence[Event]:
        with self._event_lock:
            return [event for event in self._events if event.type == type]

    def get_metrics(self):
        return list(map(lambda x: x.data, self.get_events("metrics")))

    def get_scores(self, key: str):
        return list(map(lambda e: e.data[key], self.get_events("metrics")))

    def _create_event(self, type, data=None, sample_id=None):
        if sample_id is None:
            sample_id = self.current_sample_id()
        if sample_id is None:
            raise ValueError("No sample_id set! Either pass it in or use as_default_recorder!")

        return Event(
            run_id=self.run_spec.run_id,
            event_id=len(self._events),
            type=type,
            sample_id=sample_id,
            data=data,
            created_by=self.run_spec.created_by,
            created_at=str(datetime.now(timezone.utc)),
        )

    def _flush_events_internal(self, events_to_write: Sequence[Event]):
        pass

    def flush_events(self):
        with self._event_lock:
            if len(self._events) == self._written_events:
                return
            events_to_write = self._events[self._written_events :]
            self._written_events = len(self._events)
            self._flushes_started += 1
        self._flush_events_internal(events_to_write)

    def record_event(self, type, data=None, sample_id=None):
        if sample_id is None:
            sample_id = self.current_sample_id()
        if sample_id is None:
            raise ValueError("No sample_id set! Either pass it in or use as_default_recorder!")

        if self.is_paused(sample_id):
            return
        with self._event_lock:
            event = Event(
                run_id=self.run_spec.run_id,
                event_id=len(self._events),
                type=type,
                sample_id=sample_id,
                data=data,
                created_by=self.run_spec.created_by,
                created_at=str(datetime.now(timezone.utc)),
            )
            self._events.append(event)
            if (
                self._flushes_done < self._flushes_started
                or len(self._events) < self._written_events + MIN_FLUSH_EVENTS
                or time.time() < self._last_flush_time + MIN_FLUSH_SECONDS
            ):
                return
            events_to_write = self._events[self._written_events :]
            self._written_events = len(self._events)
            self._flushes_started += 1
            self._flush_events_internal(events_to_write)

    def record_match(self, correct: bool, *, expected=None, picked=None, sample_id=None, **extra):
        assert isinstance(
            correct, bool
        ), f"correct must be a bool, but was a {type(correct)}: {correct}"

        if isinstance(expected, list) and len(expected) == 1:
            expected = expected[0]
        data = {
            "correct": bool(correct),
            "expected": expected,
            "picked": picked,
            **extra,
        }
        self.record_event("match", data, sample_id=sample_id)

    def record_embedding(self, prompt, embedding_type, sample_id=None, **extra):
        data = {
            "prompt": prompt,
            "embedding_type": embedding_type,
            **extra,
        }
        self.record_event("embedding", data, sample_id=sample_id)

    def record_sampling(self, prompt, sampled, sample_id=None, **extra):
        data = {
            "prompt": prompt,
            "sampled": sampled,
            **extra,
        }
        self.record_event("sampling", data, sample_id=sample_id)

    def record_cond_logp(self, prompt, completion, logp, sample_id=None, **extra):
        data = {
            "prompt": prompt,
            "completion": completion,
            "logp": logp,
            **extra,
        }
        self.record_event("cond_logp", data, sample_id=sample_id)

    def record_pick_option(self, prompt, options, picked, sample_id=None, **extra):
        data = {
            "prompt": prompt,
            "options": options,
            "picked": picked,
            **extra,
        }
        self.record_event("pick_option", data, sample_id=sample_id)

    def record_raw(self, data):
        self.record_event("raw_sample", data)

    def record_metrics(self, **kwargs):
        self.record_event("metrics", kwargs)

    def record_error(self, msg: str, error: Exception, **kwargs):
        data = {
            "type": type(error).__name__,
            "message": str(error),
        }
        data.update(kwargs)
        self.record_event("error", data)

    def record_extra(self, data, sample_id=None):
        self.record_event("extra", data, sample_id=sample_id)

    def record_final_report(self, final_report: Any):
        logging.info(f"Final report: {final_report}. Not writing anywhere.")


def _green(str):
    return f"\033[1;32m{str}\033[0m"


def _red(str):
    return f"\033[1;31m{str}\033[0m"


class DummyRecorder(RecorderBase):
    """
    A "recorder" which only logs certain events to the console.
    Can be used by passing `--dry-run` when invoking `oaieval`.
    """

    def __init__(self, run_spec: RunSpec, log: bool = True):
        super().__init__(run_spec)
        self.log = log

    def record_event(self, type, data, sample_id=None):
        from evals.registry import registry

        if self.run_spec is None:
            return

        base_eval_spec = registry.get_base_eval(self.run_spec.base_eval)
        if base_eval_spec and len(base_eval_spec.metrics) >= 1:
            primary_metric = base_eval_spec.metrics[0]
        else:
            primary_metric = "accuracy"

        with self._event_lock:
            event = self._create_event(type, data)
            self._events.append(event)

        msg = f"Not recording event: {event}"

        if type == "match":
            accuracy_good = (
                primary_metric == "accuracy" or primary_metric.startswith("pass@")
            ) and (data.get("correct", False) or data.get("accuracy", 0) > 0.5)
            f1_score_good = primary_metric == "f1_score" and data.get("f1_score", 0) > 0.5
            if accuracy_good or f1_score_good:
                msg = _green(msg)
            else:
                msg = _red(msg)

        if self.log:
            logging.info(msg)


class LocalRecorder(RecorderBase):
    """
    A recorder which logs events to the specified JSON file.
    This is the default recorder used by `oaieval`.
    """

    def __init__(self, log_path: Optional[str], run_spec: RunSpec):
        super().__init__(run_spec)
        self.event_file_path = log_path
        if log_path is not None:
            with bf.BlobFile(log_path, "wb") as f:
                f.write((jsondumps({"spec": dataclasses.asdict(run_spec)}) + "\n").encode("utf-8"))

    def _flush_events_internal(self, events_to_write: Sequence[Event]):
        start = time.time()
        try:
            lines = [jsondumps(event) + "\n" for event in events_to_write]
        except TypeError as e:
            logger.error(f"Failed to serialize events: {events_to_write}")
            raise e

        with bf.BlobFile(self.event_file_path, "ab") as f:
            f.write(b"".join([l.encode("utf-8") for l in lines]))

        logger.info(
            f"Logged {len(lines)} rows of events to {self.event_file_path}: insert_time={t(time.time()-start)}"
        )

        self._last_flush_time = time.time()
        self._flushes_done += 1

    def record_final_report(self, final_report: Any):
        with bf.BlobFile(self.event_file_path, "ab") as f:
            f.write((jsondumps({"final_report": final_report}) + "\n").encode("utf-8"))

        logging.info(f"Final report: {final_report}. Logged to {self.event_file_path}")


class Recorder(RecorderBase):
    """
    A recorder which logs events to Snowflake.
    Can be used by passing `--no-local-run` when invoking `oaieval`.
    """

    def __init__(
        self,
        log_path: Optional[str],
        run_spec: RunSpec,
        snowflake_connection: Optional[SnowflakeConnection] = None,
    ) -> None:
        super().__init__(run_spec)
        self.event_file_path = log_path
        self._writing_lock = threading.Lock()

        if snowflake_connection is None:
            snowflake_connection = SnowflakeConnection()
        self._conn = snowflake_connection

        if log_path is not None:
            with bf.BlobFile(log_path, "wb") as f:
                f.write((jsondumps({"spec": dataclasses.asdict(run_spec)}) + "\n").encode("utf-8"))

        query = """
            INSERT ALL INTO runs (run_id, model_name, eval_name, base_eval, split, run_config, settings, created_by, created_at)
            VALUES (%(run_id)s, %(model_name)s, %(eval_name)s, %(base_eval)s, %(split)s, run_config, settings, %(created_by)s, %(created_at)s)
            SELECT PARSE_JSON(%(run_config)s) AS run_config, PARSE_JSON(%(settings)s) AS settings
        """
        self._conn.robust_query(
            command=query,
            params={
                "run_id": run_spec.run_id,
                # TODO: model_name -> completion_fns
                "model_name": jsondumps(dict(completions=run_spec.completion_fns)),
                "eval_name": run_spec.eval_name,
                "base_eval": run_spec.base_eval,
                "split": run_spec.split,
                "run_config": jsondumps(run_spec.run_config),
                "settings": jsondumps(run_spec.run_config.get("initial_settings", {})),
                "created_by": run_spec.created_by,
                "created_at": run_spec.created_at,
            },
        )
        atexit.register(self.flush_events)

    def _flush_events_internal(self, events_to_write: Sequence[Event]):
        with self._writing_lock:
            try:
                lines = [jsondumps(event) + "\n" for event in events_to_write]
            except TypeError as e:
                logger.error(f"Failed to serialize events: {events_to_write}")
                raise e
            idx_l = 0
            while idx_l < len(events_to_write):
                total_bytes = 0
                idx_r = idx_l
                while (
                    idx_r < len(events_to_write)
                    and total_bytes + len(lines[idx_r]) < MAX_SNOWFLAKE_BYTES
                ):
                    total_bytes += len(lines[idx_r])
                    idx_r += 1
                assert idx_r > idx_l
                start = time.time()
                buffer = [
                    (
                        event.run_id,
                        event.event_id,
                        event.sample_id,
                        event.type,
                        jsondumps(event.data),
                        event.created_by,
                        event.created_at,
                    )
                    for event in events_to_write[idx_l:idx_r]
                ]
                query = """
                INSERT INTO events (run_id, event_id, sample_id, type, data, created_by, created_at)
                SELECT Column1 AS run_id, Column2 as event_id, Column3 AS sample_id, Column4 AS type, PARSE_JSON(Column5) AS data, Column6 AS created_by, Column7 AS created_at
                FROM VALUES(%s, %s, %s, %s, %s, %s, %s)
                """
                self._conn.robust_query(command=query, seqparams=buffer, many=True)
                logger.info(
                    f"Logged {len(buffer)} rows of events to Snowflake: insert_time={t(time.time()-start)}"
                )
                idx_l = idx_r

            with bf.BlobFile(self.event_file_path, "ab") as f:
                f.write(b"".join([l.encode("utf-8") for l in lines]))
            self._last_flush_time = time.time()
            self._flushes_done += 1

    def record_final_report(self, final_report: Any):
        with self._writing_lock:
            with bf.BlobFile(self.event_file_path, "ab") as f:
                f.write((jsondumps({"final_report": final_report}) + "\n").encode("utf-8"))
            query = """
                UPDATE runs
                SET final_report = PARSE_JSON(%(final_report)s)
                WHERE run_id = %(run_id)s
            """
            self._conn.robust_query(
                command=query,
                params={
                    "run_id": self.run_spec.run_id,
                    "final_report": jsondumps(final_report),
                },
            )

    def record_event(self, type, data=None, sample_id=None):
        # try to serialize data so we fail early!
        _ = jsondumps(data)
        return super().record_event(type, data, sample_id)


#########################################################################
### Helper methods which use the thread local global default recorder ###
#########################################################################


def current_sample_id() -> str:
    return default_recorder().current_sample_id


def record_match(correct: bool, *, expected=None, picked=None, **extra):
    return default_recorder().record_match(correct, expected=expected, picked=picked, **extra)


def record_embedding(prompt, embedding_type, **extra):
    return default_recorder().record_embedding(prompt, embedding_type, **extra)


def record_sampling(prompt, sampled, **extra):
    return default_recorder().record_sampling(prompt, sampled, **extra)


def record_cond_logp(prompt, completion, logp, **extra):
    return default_recorder().record_cond_logp(prompt, completion, logp, **extra)


def record_pick_option(prompt, options, picked, **extra):
    return default_recorder().record_pick_option(prompt, options, picked, **extra)


def record_raw(data):
    return default_recorder().record_raw(data)


def record_metrics(**extra):
    return default_recorder().record_metrics(**extra)


def record_error(msg: str, error: Exception = None, **extra):
    return default_recorder().record_error(msg, error, **extra)


def record_extra(data):
    return default_recorder().record_extra(data)


def record_event(type, data=None, sample_id=None):
    return default_recorder().record_event(type, data, sample_id)


def pause():
    return default_recorder().pause()


def unpause():
    return default_recorder().unpause()
