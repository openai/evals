from typing import Any, cast


class RecordingRegistry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def make_completion_fn(self, name: str, **kwargs: Any) -> object:
        self.calls.append((name, kwargs))
        return object()


def test_cot_completion_fn_forwards_constructor_kwargs(monkeypatch: Any) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    from evals.completion_fns.cot import ChainOfThoughtCompletionFn

    registry = RecordingRegistry()

    ChainOfThoughtCompletionFn(
        cot_completion_fn="reasoner",
        extract_completion_fn="extractor",
        registry=cast(Any, registry),
        temperature=0.25,
        custom_option="value",
    )

    assert registry.calls == [
        ("reasoner", {"temperature": 0.25, "custom_option": "value"}),
        ("extractor", {"temperature": 0.25, "custom_option": "value"}),
    ]
