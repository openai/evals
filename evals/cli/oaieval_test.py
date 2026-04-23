from openai.types.completion_usage import CompletionTokensDetails, CompletionUsage, PromptTokensDetails

from evals.base import RunSpec
from evals.cli.oaieval import add_token_usage_to_result
from evals.record import DummyRecorder


def test_add_token_usage_to_result_flattens_nested_usage_details() -> None:
    spec = RunSpec(
        completion_fns=[""],
        eval_name="",
        base_eval="",
        split="",
        run_config={},
        created_by="",
        run_id="",
        created_at="",
    )
    recorder = DummyRecorder(spec)

    with recorder.as_default_recorder("sample-1"):
        recorder.record_sampling(
            prompt="prompt-1",
            sampled="answer-1",
            usage=CompletionUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                completion_tokens_details=CompletionTokensDetails(reasoning_tokens=2),
                prompt_tokens_details=PromptTokensDetails(cached_tokens=4),
            ),
        )

    with recorder.as_default_recorder("sample-2"):
        recorder.record_sampling(
            prompt="prompt-2",
            sampled="answer-2",
            usage=CompletionUsage(
                prompt_tokens=7,
                completion_tokens=6,
                total_tokens=13,
                completion_tokens_details=CompletionTokensDetails(reasoning_tokens=3),
                prompt_tokens_details=PromptTokensDetails(cached_tokens=1),
            ),
        )

    result: dict[str, int] = {}
    add_token_usage_to_result(result, recorder)

    assert result["usage_prompt_tokens"] == 17
    assert result["usage_completion_tokens"] == 11
    assert result["usage_total_tokens"] == 28
    assert result["usage_completion_tokens_details_reasoning_tokens"] == 5
    assert result["usage_prompt_tokens_details_cached_tokens"] == 5
