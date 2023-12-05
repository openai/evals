import uuid
from typing import Callable, Optional, Union

import backoff
from openai import InternalServerError

from evals.api import CompletionFn, CompletionResult
from evals.elsuite.make_me_say.core import Game, Message, Player
from evals.elsuite.make_me_say.defaults import (
    DEFAULT_FEEDBACK_MSG_FN,
    DEFAULT_INTERNAL_MONOLOGUE_END_TOKEN,
    DEFAULT_INTERNAL_MONOLOGUE_START_TOKEN,
    DEFAULT_MANIPULATEE_PREAMBLE_FN,
    DEFAULT_MANIPULATOR_PREAMBLE_FN,
    DEFAULT_MAX_TURNS,
    DEFAULT_PERIODIC_MSG_FN,
)


@backoff.on_exception(backoff.constant, InternalServerError, max_tries=3)
def run(
    codeword: str,
    manipulator_completion_fn: CompletionFn,
    manipulatee_completion_fn: CompletionFn,
    judge_completion_fn: CompletionFn,
    max_turns: int = DEFAULT_MAX_TURNS,
    manipulator_preamble_fn: Callable[[Game], str] = DEFAULT_MANIPULATOR_PREAMBLE_FN,
    manipulatee_preamble_fn: Callable[[Game], str] = DEFAULT_MANIPULATEE_PREAMBLE_FN,
    internal_monologue_start_token: Optional[str] = DEFAULT_INTERNAL_MONOLOGUE_START_TOKEN,
    internal_monologue_end_token: Optional[str] = DEFAULT_INTERNAL_MONOLOGUE_END_TOKEN,
    periodic_msg_fn: Optional[Callable[[Game], str]] = DEFAULT_PERIODIC_MSG_FN,
    feedback_msg_fn: Callable[[Game], str] = DEFAULT_FEEDBACK_MSG_FN,
) -> Game:
    manipulator = Player(
        id=uuid.uuid4().hex,
        is_model=True,
        completion_fn=manipulator_completion_fn,
    )

    manipulatee = Player(
        id=uuid.uuid4().hex,
        is_model=True,
        completion_fn=manipulatee_completion_fn,
    )

    judge = Player(
        id=uuid.uuid4().hex,
        is_model=True,
        completion_fn=judge_completion_fn,
    )

    game = Game(
        manipulator=manipulator,
        manipulatee=manipulatee,
        judge=judge,
        current_player=manipulatee,
        messages=(),
        word_list=(codeword,),
        max_turns=max_turns,
        internal_monologue_start_token=internal_monologue_start_token,
        internal_monologue_end_token=internal_monologue_end_token,
    )

    manipulator_preamble = manipulator_preamble_fn(game)
    manipulatee_preamble = manipulatee_preamble_fn(game)

    assert isinstance(manipulator_preamble, str)
    assert isinstance(manipulatee_preamble, str)

    game = game.add_message(
        Message(
            role="system",
            content=manipulator_preamble,
            name=game.manipulator.id,
        ),
    )

    game = game.add_message(
        Message(
            role="system",
            content=manipulatee_preamble,
            name=game.manipulatee.id,
        )
    )

    for _ in range(game.max_turns):
        if periodic_msg_fn and game.current_player == game.manipulator:
            game = game.add_message(
                Message(
                    role="system",
                    content=periodic_msg_fn(game),
                    name=game.manipulator.id,
                )
            )

        game = _create_response(game)
        game = game.swap_players()

    game = game.add_message(
        Message(
            role="system",
            content=feedback_msg_fn(game),
            name=game.manipulatee.id,
        )
    )

    game = game.set_current_player(game.manipulatee)
    game = _create_response(game)

    return game


def _create_response(game: Game) -> Game:
    # pre-conditions
    assert game.current_player.is_model
    assert game.current_player.completion_fn is not None

    # body
    messages = [m.to_dict() for m in game.view_as(game.current_player)]
    response = game.current_player.completion_fn(messages)
    content = _get_content(response)
    new_game = game.add_message(
        Message(
            role="assistant",
            content=content,
            name=game.current_player.id,
        )
    )

    # post-conditions
    assert len(new_game.messages) == len(game.messages) + 1, "Expected one new message."
    assert new_game.messages[:-1] == game.messages, "Expected prev msgs to be the same."

    return new_game


def _get_content(response: Union[dict, CompletionResult]) -> str:
    if hasattr(response, "get_completions"):
        completions = response.get_completions()
        assert len(completions) == 1, f"Got {len(completions)} but expected exactly one"
        return completions[0]

    return response["choices"][0]["message"]["content"]
