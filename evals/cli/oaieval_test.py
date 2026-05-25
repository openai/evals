from types import SimpleNamespace

from evals.cli.oaieval import add_token_usage_to_result


class DummyRecorder:
    def __init__(self, events):
        self.events = events

    def get_events(self, type):
        return self.events


def test_add_token_usage_to_result_skips_non_numeric_usage_fields():
    result = {}
    recorder = DummyRecorder(
        [
            SimpleNamespace(
                data={
                    "usage": {
                        "completion_tokens": 14,
                        "prompt_tokens": 80,
                        "total_tokens": 94,
                        "completion_tokens_details": object(),
                        "prompt_tokens_details": object(),
                    }
                }
            )
        ]
    )

    add_token_usage_to_result(result, recorder)

    assert result == {
        "usage_completion_tokens": 14,
        "usage_prompt_tokens": 80,
        "usage_total_tokens": 94,
    }
