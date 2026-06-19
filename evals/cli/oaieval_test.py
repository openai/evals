from openai.types.completion_usage import CompletionTokensDetails, CompletionUsage

from evals.cli.oaieval import add_token_usage_to_result


class Event:
    def __init__(self, data):
        self.data = data


class Recorder:
    def __init__(self, events):
        self.events = events

    def get_events(self, event_type):
        assert event_type == "sampling"
        return self.events


def test_add_token_usage_to_result_flattens_structured_usage_details():
    result = {}
    recorder = Recorder(
        [
            Event(
                {
                    "usage": CompletionUsage(
                        completion_tokens=3,
                        prompt_tokens=5,
                        total_tokens=8,
                        completion_tokens_details=CompletionTokensDetails(reasoning_tokens=2),
                    )
                }
            ),
            Event(
                {
                    "usage": CompletionUsage(
                        completion_tokens=4,
                        prompt_tokens=6,
                        total_tokens=10,
                        completion_tokens_details=CompletionTokensDetails(reasoning_tokens=3),
                    )
                }
            ),
        ]
    )

    add_token_usage_to_result(result, recorder)

    assert result["usage_completion_tokens"] == 7
    assert result["usage_prompt_tokens"] == 11
    assert result["usage_total_tokens"] == 18
    assert result["usage_completion_tokens_details_reasoning_tokens"] == 5
