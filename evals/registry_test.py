import pytest

from evals import OpenAIChatCompletionFn, OpenAICompletionFn
from evals.registry import Registry, is_chat_model, n_ctx_from_model_name, openai_model_endpoint


def test_n_ctx_from_model_name():
    assert n_ctx_from_model_name("gpt-3.5-turbo") == 4096
    assert n_ctx_from_model_name("gpt-3.5-turbo-0613") == 4096
    assert n_ctx_from_model_name("gpt-3.5-turbo-16k") == 16384
    assert n_ctx_from_model_name("gpt-3.5-turbo-16k-0613") == 16384
    assert n_ctx_from_model_name("gpt-4") == 8192
    assert n_ctx_from_model_name("gpt-4-0613") == 8192
    assert n_ctx_from_model_name("gpt-4-32k") == 32768
    assert n_ctx_from_model_name("gpt-4-32k-0613") == 32768
    assert n_ctx_from_model_name("gpt-3.5-turbo") == 4096
    assert n_ctx_from_model_name("gpt-3.5-turbo-0314") == 4096
    assert n_ctx_from_model_name("gpt-3.5-turbo-0613") == 4096
    assert n_ctx_from_model_name("gpt-3.5-turbo-16k") == 16384
    assert n_ctx_from_model_name("gpt-3.5-turbo-16k-0314") == 16384
    assert n_ctx_from_model_name("gpt-3.5-turbo-16k-0613") == 16384


@pytest.mark.parametrize(
    "model_name",
    [
        "chat-latest",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0613",
        "gpt-4",
        "gpt-4-0613",
        "gpt-4-32k",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4o-2024-11-20",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-2025-04-14",
        "gpt-4.5-preview",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-2025-08-07",
        "gpt-5.1",
        "gpt-5.1-2025-11-13",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-5.6",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "o1",
        "o1-mini",
        "o1-2024-12-17",
        "o3",
        "o3-mini",
        "o4-mini",
    ],
)
def test_modern_text_models_use_chat_completions(model_name):
    assert openai_model_endpoint(model_name) == "chat"
    assert is_chat_model(model_name)


@pytest.mark.parametrize(
    "model_name",
    [
        "ada",
        "babbage-002",
        "davinci-002",
        "text-davinci-003",
        "code-davinci-002",
        "gpt-3.5-turbo-instruct",
        "gpt-3.5-turbo-instruct-0914",
        "gpt-4-base",
    ],
)
def test_legacy_models_use_completions(model_name):
    assert openai_model_endpoint(model_name) == "completions"
    assert not is_chat_model(model_name)


@pytest.mark.parametrize(
    "model_name",
    [
        "gpt-4o-mini-transcribe",
        "gpt-4o-realtime-preview",
        "gpt-5-pro",
        "gpt-5.1-codex",
        "gpt-5.2-codex",
        "gpt-5.4-pro",
        "o1-pro",
        "o3-pro",
        "o3-deep-research",
        "o4-mini-deep-research",
    ],
)
def test_specialised_models_are_not_guessed_as_chat_or_legacy_completions(model_name):
    assert openai_model_endpoint(model_name) is None
    assert not is_chat_model(model_name)


def test_fine_tuned_model_routes_using_its_base_model():
    assert openai_model_endpoint("ft:gpt-4o-mini:acme:eval:abc123") == "chat"
    assert openai_model_endpoint("ft:davinci-002:acme:eval:abc123") == "completions"


@pytest.mark.parametrize(
    "model_name", ["gpt-4o-mini", "o3-mini", "gpt-5-mini", "gpt-5.6-terra"]
)
def test_registry_routes_modern_models_to_chat_completion_fn(model_name):
    registry = Registry(registry_paths=[])

    completion_fn = registry.make_completion_fn(model_name)

    assert isinstance(completion_fn, OpenAIChatCompletionFn)
    assert completion_fn.model == model_name


def test_registry_routes_known_legacy_model_to_completion_fn():
    registry = Registry(registry_paths=[])

    completion_fn = registry.make_completion_fn("davinci-002")

    assert isinstance(completion_fn, OpenAICompletionFn)
    assert completion_fn.model == "davinci-002"


def test_api_listed_unknown_model_is_not_silently_sent_to_legacy_completions():
    registry = Registry(registry_paths=[])
    registry.__dict__["api_model_ids"] = ["gpt-5.4-pro"]

    with pytest.raises(ValueError, match="cannot safely infer a supported endpoint"):
        registry.make_completion_fn("gpt-5.4-pro")
