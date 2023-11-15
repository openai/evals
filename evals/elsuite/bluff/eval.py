import logging
import os
import random
from importlib import import_module
from typing import Optional, Union

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from openai.error import InvalidRequestError

import evals
from evals.api import CompletionFn
from evals.elsuite.bluff.bluff.game import Game
from evals.elsuite.bluff.bluff.players import Player
from evals.elsuite.bluff.solver_player import SolverPlayer
from evals.eval import SolverEval
from evals.solvers.human_cli_solver import HumanCliSolver
from evals.solvers.solver import Solver

registry = evals.registry.Registry()
logger = logging.getLogger(__name__)


class BluffEval(SolverEval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        opponent: str,
        n_samples: int,
        n_rounds: int = 10,
        seed: Optional[int] = None,
        *args,
        **kwargs,
    ):
        super().__init__(completion_fns, seed=seed, *args, **kwargs)

        self.opponent_name = opponent
        self.n_samples = n_samples
        self.num_rounds = n_rounds

    def eval_sample(self, solver: Solver, sample_ix: int, rng: random.Random):
        game = Game(self.num_rounds, starting_player=sample_ix % 2, rng=rng)
        player_0 = SolverPlayer(game, solver)
        player_1 = self._create_opponent(game)

        #   Separate rng so that:
        #   *   As long as our play doesn't change between runs, neither does bot's
        #   *   If we change our play between runs, we'll still play the same hands
        #       (because our decisions have no impact on the main rng)
        player_1.rng = np.random.default_rng(rng.randint(0, 10**9))

        info = {
            "sample_ix": sample_ix,
            "player_0": self._get_player_info(player_0),
            "player_1": self._get_player_info(player_1),
        }

        try:
            game.play()
            evals.record.record_metrics(
                **info,
                **self._get_game_metrics(game),
            )
        except InvalidRequestError as e:
            if str(e).startswith("This model's maximum context length is"):
                logger.warning(
                    f"Game exceeded the context window - sample {sample_ix} will be ignored. Consider decreasing n_rounds."
                )
            else:
                raise

    def _get_player_info(self, player: Player) -> str:
        if isinstance(player, SolverPlayer):
            return player.solver.name
        else:
            return type(player).__name__

    def run(self, recorder: evals.record.Recorder) -> dict[str, Union[float, int]]:
        samples = list(range(self.n_samples))
        self.eval_all_samples(recorder, samples)
        metrics = recorder.get_metrics()

        #   1.  Per-round winrate
        winners = [m["wins"] for m in metrics]
        player_0_wins = sum(
            sum(winner == 0 for winner in round_winners) for round_winners in winners
        )
        player_1_wins = sum(
            sum(winner == 1 for winner in round_winners) for round_winners in winners
        )
        round_cnt = player_0_wins + player_1_wins

        #   2.  Per-round-ix winrate (e.g. did we learn from early rounds?)
        #   Note: we don't use self.n_samples because some games might have exceeded the context window length
        num_games = len(metrics)

        player_0_per_round_wins = [0] * self.num_rounds
        player_1_per_round_wins = [0] * self.num_rounds
        for round_winners in winners:
            for round_ix, winner in enumerate(round_winners):
                player_0_per_round_wins[round_ix] += int(winner == 0)
                player_1_per_round_wins[round_ix] += int(winner == 1)

        player_0_per_round_win_ratio = [wins / num_games for wins in player_0_per_round_wins]

        #   3.  Tests for the round_ix winrate changes
        data = pd.DataFrame(
            [
                list(range(self.num_rounds)),
                player_0_per_round_win_ratio,
            ],
            ["round_ix", "wins"],
        ).transpose()

        results = smf.ols("wins ~ round_ix", data=data).fit()
        print(results.summary())

        #   4.  Additional data - how rounds ended
        player_0_bid_won = 0
        player_0_bid_lost = 0
        player_0_called_bluff_won = 0
        player_0_called_bluff_lost = 0
        for game_data in metrics:
            round_data = zip(game_data["wins"], game_data["who_called_bluff"])
            for winner, who_called_bluff in round_data:
                if winner == 0 and who_called_bluff == 0:
                    player_0_called_bluff_won += 1
                elif winner == 0 and who_called_bluff == 1:
                    player_0_bid_won += 1
                elif winner == 1 and who_called_bluff == 0:
                    player_0_called_bluff_lost += 1
                elif winner == 1 and who_called_bluff == 1:
                    player_0_bid_lost += 1

        return {
            "valid_samples": num_games,
            "too_long_games": self.n_samples - num_games,
            "player_0": metrics[0]["player_0"],
            "player_1": metrics[0]["player_1"],
            "player_0_wins": player_0_wins,
            "player_1_wins": player_1_wins,
            "player_0_win_ratio": player_0_wins / round_cnt,
            "player_0_per_round_wins": player_0_per_round_wins,
            "player_1_per_round_wins": player_1_per_round_wins,
            "player_0_round_ix_coef": results.params["round_ix"],
            "player_0_round_ix_pvalue": results.pvalues["round_ix"],
            "player_0_bid_won": player_0_bid_won,
            "player_0_bid_lost": player_0_bid_lost,
            "player_0_called_bluff_won": player_0_called_bluff_won,
            "player_0_called_bluff_lost": player_0_called_bluff_lost,
        }

    def _get_game_metrics(self, game: Game) -> dict:
        rounds = [round for round in game.rounds if round.finished]
        wins = [round.winner for round in rounds]
        who_bid_last = [round.moves[-1][0] for round in rounds]
        who_called_bluff = [1 - player for player in who_bid_last]

        result = {
            "wins": wins,
            "who_called_bluff": who_called_bluff,
        }
        return result

    def _create_opponent(self, game: Game) -> Player:
        if self.opponent_name == "human_cli":
            return self._create_human_player(game)
        else:
            try:
                return self._create_solver_player(game, self.opponent_name)
            except ValueError:
                try:
                    return self._create_bot_player(game, self.opponent_name)
                except Exception:
                    raise ValueError(
                        f"Can't parse opponent {self.opponent_name}. Pass either a bot class or a solver name."
                    )

    @staticmethod
    def _create_human_player(game: Game) -> Player:
        if os.environ.get("EVALS_SEQUENTIAL") != "1":
            raise ValueError("human_cli player is available only with EVALS_SEQUENTIAL=1")

        solver = HumanCliSolver()
        return SolverPlayer(game, solver)

    @staticmethod
    def _create_solver_player(game: Game, solver_name: str) -> Player:
        #   This logger.disabled thing prevents messages saying that completion_fn was
        #   not found (because they are usually emitted )
        evals.registry.logger.disabled = True
        solver = registry.make_completion_fn(solver_name)
        evals.registry.logger.disabled = False
        return SolverPlayer(game, solver)

    @staticmethod
    def _create_bot_player(game: Game, module_and_class: str) -> Player:
        module_name, class_name = module_and_class.split(":")
        module = import_module(module_name)
        bot_class = getattr(module, class_name)
        return bot_class(game)
