from evals.registry import is_chat_model, n_ctx_from_model_name


def test_n_ctx_from_model_name():
    assert n_ctx_from_model_name("gpt-3.5-turbo") == 4096
    assert n_ctx_from_model_name("gpt-3.5-turbo-0613") == 4096
    assert n_ctx_from_model_name("gpt-3.5-turbo-16k") == 16384
    assert n_ctx_from_model_name("gpt-3.5-turbo-16k-0613") == 16384
    assert n_ctx_from_model_name("gpt-4o") == 128_000
    assert n_ctx_from_model_name("o1-preview") == 128_000
    assert n_ctx_from_model_name("o1-mini") == 128_000
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
    assert is_chat_model("gpt-4o")
    assert is_chat_model("o1-preview")
    assert is_chat_model("o1-mini")
    assert not is_chat_model("text-davinci-003")
    assert not is_chat_model("gpt4-base")
    assert not is_chat_model("code-davinci-002")
