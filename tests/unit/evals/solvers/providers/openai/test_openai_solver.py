import json
from copy import deepcopy

import httpx
import pytest
import tiktoken
from openai import OpenAI

import evals.completion_fns.openai as completion_module
from evals.base import RunSpec
from evals.record import RecorderBase
from evals.solvers.providers.openai.openai_solver import OpenAISolver
from evals.task_state import Message, TaskState


@pytest.fixture
def byte_encoding(monkeypatch):
    encoding = tiktoken.Encoding(
        name="synthetic-valid-answer-bytes",
        pat_str=r"[\s\S]",
        mergeable_ranks={bytes([i]): i for i in range(256)},
        special_tokens={},
    )
    monkeypatch.setattr(tiktoken, "encoding_for_model", lambda _model: encoding)
    return encoding


@pytest.mark.parametrize("model", ["gpt-4", "text-davinci-003"])
@pytest.mark.parametrize("extra_kind", ["missing", "empty", "null", "existing"])
def test_valid_answers_reach_the_serialized_request(model, extra_kind, byte_encoding, monkeypatch):
    options = {"model": model}
    if extra_kind != "missing":
        options["extra_options"] = None if extra_kind == "null" else {}
    if extra_kind == "existing":
        options["extra_options"] = {"temperature": 0.25, "max_tokens": 7, "logit_bias": {3: -20}}
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        choice = {"index": 0, "finish_reason": "stop"}
        if model == "gpt-4":
            choice["message"] = {"role": "assistant", "content": "A"}
        else:
            choice.update(text="A", logprobs=None)
        return httpx.Response(
            200,
            json={
                "id": "synthetic-completion",
                "object": "chat.completion" if model == "gpt-4" else "text_completion",
                "created": 0,
                "model": model,
                "choices": [choice],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    with OpenAI(
        api_key="offline-test",
        base_url="https://example.test/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    ) as client:
        monkeypatch.setattr(completion_module, "OpenAI", lambda **_kwargs: client)
        solver = OpenAISolver(completion_fn_options=options, valid_answers=["A", "B"])
        recorder = RecorderBase(
            RunSpec(
                completion_fns=["synthetic"],
                eval_name="test.dev.v0",
                base_eval="test",
                split="dev",
                run_config={},
                created_by="test",
            )
        )
        with recorder.as_default_recorder("test.dev.0"):
            result = solver(
                TaskState("Select an answer.", [Message("user", "A or B?")]), max_tokens=1
            )
    assert result.output == "A"
    assert len(requests) == 1
    path, body = requests[0]
    assert path == ("/v1/chat/completions" if model == "gpt-4" else "/v1/completions")
    assert body["logit_bias"] == {"65": 100, "66": 100}
    assert body["model"] == model
    assert body["max_tokens"] == (7 if extra_kind == "existing" else 1)
    if extra_kind == "existing":
        assert body["temperature"] == 0.25
    assert len(recorder.get_events("sampling")) == 1


@pytest.mark.parametrize("valid_answers", [None, []])
@pytest.mark.parametrize("with_options", [False, True])
def test_absent_valid_answers_do_not_add_bias_or_create_options(
    valid_answers, with_options, monkeypatch
):
    options = {"model": "gpt-4"}
    if with_options:
        options["extra_options"] = {"temperature": 0.25, "logit_bias": {3: -20}}
    original = deepcopy(options)

    def unexpected_bias(*_args):
        raise AssertionError("disabled valid_answers must not tokenize")

    monkeypatch.setattr(OpenAISolver, "_make_logit_bias", unexpected_bias)
    solver = OpenAISolver(completion_fn_options=options, valid_answers=valid_answers)
    assert solver.completion_fn_options == original
    assert options == original


def test_failed_answer_validation_leaves_missing_options_unchanged(byte_encoding):
    options = {"model": "gpt-4"}
    with pytest.raises(ValueError, match="single token"):
        OpenAISolver(completion_fn_options=options, valid_answers=["AB"])
    assert options == {"model": "gpt-4"}


def test_existing_options_mapping_is_preserved(byte_encoding):
    extra_options = {"temperature": 0.25}
    solver = OpenAISolver(
        completion_fn_options={"model": "gpt-4", "extra_options": extra_options},
        valid_answers=["A", "A", "B"],
    )
    assert solver.completion_fn_options["extra_options"] is extra_options
    assert extra_options == {"temperature": 0.25, "logit_bias": {65: 100, 66: 100}}
