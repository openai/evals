from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from evals.registry import Registry, is_chat_model, n_ctx_from_model_name


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


def test_is_chat_model():
    assert is_chat_model("gpt-3.5-turbo")
    assert is_chat_model("gpt-3.5-turbo-0613")
    assert is_chat_model("gpt-3.5-turbo-16k")
    assert is_chat_model("gpt-3.5-turbo-16k-0613")
    assert is_chat_model("gpt-4")
    assert is_chat_model("gpt-4-0613")
    assert is_chat_model("gpt-4-32k")
    assert is_chat_model("gpt-4-32k-0613")
    assert not is_chat_model("text-davinci-003")
    assert not is_chat_model("gpt4-base")
    assert not is_chat_model("code-davinci-002")


def test_api_model_ids_initializes_openai_client_lazily():
    registry = Registry(registry_paths=[])
    fake_client = MagicMock()
    fake_client.models.list.return_value = SimpleNamespace(
        data=[SimpleNamespace(id="model-a"), SimpleNamespace(id="model-b")]
    )

    with patch("evals.registry.OpenAI", return_value=fake_client) as openai_client:
        openai_client.assert_not_called()

        assert registry.api_model_ids == ["model-a", "model-b"]
        openai_client.assert_called_once()
        fake_client.models.list.assert_called_once()

        # api_model_ids is cached; repeated routing checks do not repeat model discovery.
        assert registry.api_model_ids == ["model-a", "model-b"]
        openai_client.assert_called_once()
        fake_client.models.list.assert_called_once()
