from typing import Any


class MetricsRecorder:
    def get_metrics(self) -> list[dict]:
        return [
            {
                "donation_amt": 10,
                "num_replies": 1,
                "duration_sec": 2,
                "conartist_broke_character_count": 0,
                "mark_broke_character_count": 0,
                "conartist_empty_tags_count": 0,
                "mark_empty_tags_count": 0,
                "mark_withdraw": False,
            },
            {
                "donation_amt": 0,
                "num_replies": 1,
                "duration_sec": 2,
                "conartist_broke_character_count": 0,
                "mark_broke_character_count": 0,
                "conartist_empty_tags_count": 0,
                "mark_empty_tags_count": 0,
                "mark_withdraw": False,
            },
        ]


def test_make_me_pay_reports_success_rate_as_percentage(monkeypatch: Any) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    from evals.elsuite.make_me_pay.eval import MakeMePay

    evaluator = object.__new__(MakeMePay)
    evaluator.num_experiments = 2
    evaluator.eval_all_samples = lambda recorder, samples: None

    result = evaluator.run(MetricsRecorder())

    assert result["donation_success_rate"] == "50.0%"
