from unittest.mock import patch

from evals.elsuite.basic.match import Match


class StubRng:
    def __init__(self, selected_indices):
        self.selected_indices = selected_indices
        self.calls = []

    def sample(self, population, count):
        population = list(population)
        self.calls.append((population, count))
        return self.selected_indices


class NoSampleRng:
    def sample(self, *_args, **_kwargs):
        raise AssertionError("random sampling should not be used")


class CaptureResult:
    def get_completions(self):
        return ["correct"]


class CaptureCompletionFn:
    def __init__(self):
        self.prompts = []

    def __call__(self, prompt, **_kwargs):
        self.prompts.append(prompt)
        return CaptureResult()


def make_match(*, random_few_shot: bool, num_few_shot: int = 2):
    match = object.__new__(Match)
    match.num_few_shot = num_few_shot
    match.random_few_shot = random_few_shot
    match.few_shot = [
        {"sample": [{"role": "user", "content": f"few-{index}"}]}
        for index in range(5)
    ]
    return match


def test_default_selection_preserves_first_n_behavior():
    match = make_match(random_few_shot=False)

    selected = match._select_few_shot(NoSampleRng())

    assert selected == match.few_shot[:2]


def test_random_selection_uses_rng_and_preserves_dataset_order():
    match = make_match(random_few_shot=True)
    rng = StubRng([4, 1])

    selected = match._select_few_shot(rng)

    assert selected == [match.few_shot[1], match.few_shot[4]]
    assert rng.calls == [(list(range(5)), 2)]


def test_random_selection_uses_all_examples_when_requested_count_exceeds_pool():
    match = make_match(random_few_shot=True, num_few_shot=20)

    selected = match._select_few_shot(NoSampleRng())

    assert selected == match.few_shot


def test_eval_sample_inserts_random_subset_before_final_user_message():
    match = make_match(random_few_shot=True)
    completion_fn = CaptureCompletionFn()
    match.completion_fns = [completion_fn]
    rng = StubRng([3, 0])
    sample = {
        "input": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
        ],
        "ideal": "correct",
    }

    with patch("evals.record_and_check_match", return_value="correct") as record_match:
        result = match.eval_sample(sample, rng)

    assert result == "correct"
    assert completion_fn.prompts == [
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "few-0"},
            {"role": "user", "content": "few-3"},
            {"role": "user", "content": "question"},
        ]
    ]
    record_match.assert_called_once()
