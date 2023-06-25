from evals.registry import is_chat_model, n_ctx_from_model_name


def test_n_ctx_from_model_name():
    assert n_ctx_from_model_name("gpt-4") == 8192
    assert n_ctx_from_model_name("gpt-4-0314") == 8192
    assert n_ctx_from_model_name("gpt-4-0613") == 8192
    assert n_ctx_from_model_name("gpt-4-32k") == 32768
    assert n_ctx_from_model_name("gpt-4-32k-0314") == 32768
    assert n_ctx_from_model_name("gpt-4-32k-0613") == 32768

def test_is_chat_model():
    assert is_chat_model("gpt-3.5-turbo")
    assert is_chat_model("gpt-3.5-turbo-0314")
    assert is_chat_model("gpt-3.5-turbo-0613")
    assert is_chat_model("gpt-4")
    assert is_chat_model("gpt-4-0314")
    assert is_chat_model("gpt-4-0613")
    assert is_chat_model("gpt-4-32k")
    assert is_chat_model("gpt-4-32k-0314")
    assert is_chat_model("gpt-4-32k-0613")
    assert not is_chat_model("text-davinci-003")
