from pathlib import Path
from typing import Any, cast


class RecordingRegistry:
    def __init__(self) -> None:
        self.call: Any = None

    def make_completion_fn(self, completion_fn: str, **kwargs: Any) -> Any:
        self.call = (completion_fn, kwargs)
        return object()


def test_retrieval_completion_fn_forwards_constructor_kwargs(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    from evals.completion_fns.retrieval import RetrievalCompletionFn

    embeddings_path = tmp_path / "embeddings.csv"
    embeddings_path.write_text('text,embedding\ncontext,"[1.0, 0.0]"\n', encoding="utf-8")
    registry = RecordingRegistry()

    RetrievalCompletionFn(
        completion_fn="custom-completion",
        embeddings_and_text_path=str(embeddings_path),
        registry=cast(Any, registry),
        temperature=0.25,
        custom_option="value",
    )

    assert registry.call == (
        "custom-completion",
        {"temperature": 0.25, "custom_option": "value"},
    )
