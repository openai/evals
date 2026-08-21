from typing import Any


def test_posthoc_ctrl_recall_skips_no_control_sample_after_parse_failure(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    from evals.elsuite.identifying_variables.metrics import compute_ctrl_recall_posthoc

    metric_entries = [
        {
            "gold_answer": {
                "valid_hypothesis": True,
                "ctrl_vars": [],
            }
        },
        {
            "gold_answer": {
                "valid_hypothesis": True,
                "ctrl_vars": ["x"],
            }
        },
    ]
    sampling_entries = [
        {"sampled": ["malformed output"]},
        {
            "sampled": [
                "[@answer valid_hyp: true; independent: a; dependent: b; control: x]"
            ]
        },
    ]

    assert compute_ctrl_recall_posthoc(metric_entries, sampling_entries) == 1.0


def test_posthoc_ctrl_recall_still_penalizes_parse_failure_when_recall_is_defined(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    from evals.elsuite.identifying_variables.metrics import compute_ctrl_recall_posthoc

    metric_entries = [
        {
            "gold_answer": {
                "valid_hypothesis": True,
                "ctrl_vars": ["x"],
            }
        }
    ]
    sampling_entries = [{"sampled": ["malformed output"]}]

    assert compute_ctrl_recall_posthoc(metric_entries, sampling_entries) == 0.0
