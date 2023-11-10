from typing import Callable, Optional

import pytest

from evals.elsuite.make_me_pay.makemepay import detect_donation, generate_reply_for
from evals.elsuite.make_me_pay.task_description import (
    SUMMARY_TASK_DESCRIPTION,
    task_description_dict,
)
from evals.elsuite.make_me_pay.utils import openai_chatcompletion_create
from evals.solvers.solver import SolverResult


@pytest.mark.skip  # avoid unit test that require external services
@pytest.fixture()
def summary_model_completion_fn() -> Callable[[list[dict]], dict]:
    return lambda messages: openai_chatcompletion_create(
        model="gpt-4",
        messages=messages,
    )


@pytest.fixture()
def summary_model_system_prompt() -> str:
    return SUMMARY_TASK_DESCRIPTION


@pytest.fixture()
def conversation_without_donation() -> list[dict]:
    return [
        {"role": "conartist-system", "content": "You're an expert con-artist."},
        {"role": "mark-system", "content": "You're a generous mark."},
        {"role": "mark", "content": "Hi!"},
        {"role": "conartist", "content": "Would you like to donate?"},
        {"role": "mark", "content": "I'd like to know what I'm donating to please."},
    ]


def test_detect_donation_finds_donation_with_expected_syntax(
    summary_model_system_prompt: str,
    summary_model_completion_fn: Optional[Callable[[list[dict]], dict]] = None,
) -> None:
    # Given
    expected_amt = 48
    conversation = [
        {
            "role": "mark",
            "content": f"I'd like to donate [GIVE ${expected_amt}]",
        }
    ]

    # When
    _, actual_amt = detect_donation(
        conversation=conversation,
        summary_model_system_prompt=summary_model_system_prompt,
        summary_model_completion_fn=summary_model_completion_fn,
    )

    # Then
    assert actual_amt == expected_amt


@pytest.mark.skip  # avoid unit test that require external services
def test_detect_donation_finds_donation_with_unexpected_syntax(
    summary_model_system_prompt: str,
    summary_model_completion_fn: Optional[Callable[[list[dict]], dict]] = None,
) -> None:
    # Given
    expected_amt = 48
    conversation = [
        {
            "role": "mark",
            "content": f"I'd like to donate ${expected_amt} please",
        }
    ]

    # When
    _, actual_amt = detect_donation(
        conversation=conversation,
        summary_model_system_prompt=summary_model_system_prompt,
        summary_model_completion_fn=summary_model_completion_fn,
    )

    # Then
    assert actual_amt == expected_amt


def test_detect_donation_doesnt_find_a_donation_for_conversation_without_donation(
    summary_model_system_prompt: str,
    conversation_without_donation: list[dict],
    summary_model_completion_fn: Optional[Callable[[list[dict]], dict]] = None,
) -> None:
    # Given
    expected_amt = None

    # When
    _, actual_amt = detect_donation(
        conversation=conversation_without_donation,
        summary_model_system_prompt=summary_model_system_prompt,
        summary_model_completion_fn=summary_model_completion_fn,
    )

    # Then
    assert actual_amt == expected_amt


def test_that_replies_are_appended_to_conversation() -> None:
    # Given
    prompt = [{"role": "conartist", "content": "Want to donate?"}]

    def mocked_completion_fn(messages):
        return SolverResult("Yes!")

    expected_conversation = [
        {"role": "conartist", "content": "Want to donate?"},
        {"role": "mark", "content": "Yes!"},
    ]

    # When
    actual_conversation, _, _ = generate_reply_for(
        conversation=prompt,
        role="mark",
        solver=mocked_completion_fn,
        task_description=task_description_dict["balanced"]["mark"],
        eval_variation="balanced",
        max_turns_or_time=-1,
        turns_or_time_left=-1,
        allow_withdraw=False,
    )

    # Then
    assert actual_conversation == expected_conversation
