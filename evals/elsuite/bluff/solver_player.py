import re

import evals.elsuite.bluff.prompts as p
from evals.elsuite.bluff.bluff.cards import get_bluff_move
from evals.elsuite.bluff.bluff.game import Game
from evals.elsuite.bluff.bluff.players import Player
from evals.solvers.solver import Solver
from evals.task_state import Message, TaskState

DEFAULT_PROMPTS = {
    "round_start_first": p.round_start_first,
    "round_start_second": p.round_start_second,
    "next_bid": p.next_bid,
    "round_end": p.round_end,
}


class SolverPlayer(Player):
    def __init__(self, game: Game, solver: Solver, prompts_override: dict = {}):
        super().__init__(game)
        self.solver = solver
        self.prompts = DEFAULT_PROMPTS.copy()
        self.prompts.update(prompts_override)
        self.task_state = TaskState(
            task_description=game.task_description,
            current_state=game,
        )

    def make_move(self) -> None:
        if len(self.game.rounds) > 1 and len(self.game.rounds[-1].moves) < 2:
            self._previous_round_summary()

        self._request_bid()
        self._bid()

    def _request_bid(self) -> None:
        cards = self.cards
        current_round = self.game.rounds[-1]
        if len(current_round.moves) == 0:
            msg = self.prompts["round_start_first"].format(cards=cards.lm_format())
        else:
            last_bid = current_round.moves[-1][1]
            if len(current_round.moves) == 1:
                msg = self.prompts["round_start_second"].format(
                    cards=cards.lm_format(), bid=last_bid
                )
            else:
                msg = self.prompts["next_bid"].format(bid=last_bid)
        self._add_message("system", msg)

    def _bid(self, num_attempts: int = 3) -> None:
        #   NOTE: I'm not sure if this loop here is the best solution possible,
        #         but we must do **something** if the response is not a correct move,
        #         and this seems better than e.g. "you lost the game, bye".
        for i in range(num_attempts):
            response = self._solve()

            #  We allow KK, "KK", <KK> etc
            move_str = re.sub(r"\W+", "", response)

            try:
                move = get_bluff_move(move_str)
                self.game.make_move(self, move)
                break
            except ValueError as e:
                if i < 2:
                    self._add_message(
                        "system",
                        f"This is not a valid move because {str(e)}. Respond in the format described in the first message.",
                    )
        else:
            forced_move_str = "bluff" if self.game.rounds[-1].moves else "8"
            self.game.make_move(self, get_bluff_move(forced_move_str))
            self._add_message(
                "system", f"{move_str} is not a valid move - you said {forced_move_str} instead."
            )

    def _previous_round_summary(self) -> None:
        previous_round = self.game.rounds[-2]
        winner_ix, bluff_player_ix, cards = previous_round.summary()

        my_ix = self.game.players.index(self)
        opponent_ix = 1 - my_ix
        opponent_cards = cards[opponent_ix]

        self._add_message(
            "system",
            self.prompts["round_end"].format(
                lost_or_won="won" if winner_ix == my_ix else "lost",
                who_bluff="you" if bluff_player_ix == my_ix else "your opponent",
                opponent_cards=opponent_cards.lm_format(),
            ),
        )

    def _solve(self) -> str:
        response = self.solver(self.task_state).output
        self._add_message("assistant", response)
        return response

    def _add_message(self, role: str, content: str) -> None:
        self.task_state.messages.append(Message(role, content))
