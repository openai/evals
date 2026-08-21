from evals.solvers.providers.openai.openai_solver import OpenAISolver


class StubOpenAISolver(OpenAISolver):
    def _make_logit_bias(self, valid_answers: list[str], model: str) -> dict[int, float]:
        assert valid_answers == ["A", "B"]
        assert model == "gpt-4"
        return {101: 100, 202: 100}


def _solver_with_options(completion_fn_options: dict) -> StubOpenAISolver:
    solver = object.__new__(StubOpenAISolver)
    solver.valid_answers = ["A", "B"]
    solver.completion_fn_options = completion_fn_options
    return solver


def test_valid_answers_create_missing_extra_options() -> None:
    solver = _solver_with_options({"model": "gpt-4"})

    solver._preprocess_completion_fn_options()

    assert solver.completion_fn_options == {
        "model": "gpt-4",
        "extra_options": {"logit_bias": {101: 100, 202: 100}},
    }


def test_valid_answers_preserve_existing_extra_options() -> None:
    solver = _solver_with_options(
        {"model": "gpt-4", "extra_options": {"temperature": 0.0, "max_tokens": 1}}
    )

    solver._preprocess_completion_fn_options()

    assert solver.completion_fn_options["extra_options"] == {
        "temperature": 0.0,
        "max_tokens": 1,
        "logit_bias": {101: 100, 202: 100},
    }
