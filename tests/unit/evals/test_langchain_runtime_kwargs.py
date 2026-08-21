from types import SimpleNamespace
from typing import Any


def _load_module(monkeypatch: Any):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    from evals.completion_fns import langchain_llm

    monkeypatch.setattr(langchain_llm, "record_sampling", lambda **_kwargs: None)
    return langchain_llm


def test_langchain_llm_forwards_runtime_kwargs(monkeypatch: Any) -> None:
    langchain_llm = _load_module(monkeypatch)

    class RecordingLLM:
        def __init__(self) -> None:
            self.calls = []

        def __call__(self, prompt, **kwargs):
            self.calls.append((prompt, kwargs))
            return "ok"

    model = RecordingLLM()
    completion_fn = object.__new__(langchain_llm.LangChainLLMCompletionFn)
    completion_fn.llm = model

    result = completion_fn("hello", stop=["END"], custom_option="value")

    assert model.calls == [("hello", {"stop": ["END"], "custom_option": "value"})]
    assert result.get_completions() == ["ok"]


def test_langchain_chat_model_forwards_runtime_kwargs(monkeypatch: Any) -> None:
    langchain_llm = _load_module(monkeypatch)

    class RecordingChatModel:
        def __init__(self) -> None:
            self.kwargs = None

        def __call__(self, messages, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(content="ok")

    model = RecordingChatModel()
    completion_fn = object.__new__(langchain_llm.LangChainChatModelCompletionFn)
    completion_fn.llm = model

    result = completion_fn(
        [{"role": "user", "content": "hello"}],
        stop=["END"],
        custom_option="value",
    )

    assert model.kwargs == {"stop": ["END"], "custom_option": "value"}
    assert result.get_completions() == ["ok"]
