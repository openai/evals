import logging
import random
from typing import Any, Dict, Sequence, Union

import numpy as np

import evals.metrics
from evals.elsuite.cant_do_that_anymore.chess.board import BoardController
from evals.elsuite.cant_do_that_anymore.chess.board_test import default_board_init
from evals.elsuite.cant_do_that_anymore.chess.move_variants import (
    PIECE_ID_TO_INSTANCE,
    PIECE_ID_TO_STR,
    PIECE_STR_TO_ID,
    VARIANT_PIECE_ID_TO_INSTANCE,
)
from evals.elsuite.cant_do_that_anymore.chess.notation import AlgebraicNotationParser
from evals.elsuite.cant_do_that_anymore.chess.pieces import Piece
from evals.elsuite.cant_do_that_anymore.chess.utils import (
    capturing_same_color,
    move_within_board,
    same_color_piece_at_move_start,
)
from evals.elsuite.cant_do_that_anymore.defaults import TASK_DESCRIPTION, TASK_DESCRIPTION_VARIANT
from evals.elsuite.cant_do_that_anymore.utils import (
    construct_messages,
    get_binary_avg,
    get_dataset_path,
    get_diagonal_dataset_path,
)
from evals.eval import SolverEval
from evals.record import RecorderBase
from evals.solvers.solver import Solver, SolverResult
from evals.task_state import TaskState

logger = logging.getLogger(__name__)


class CantDoThatAnymore(SolverEval):
    def __init__(
        self,
        default_model_dataset: str = "gpt-3.5-turbo-0125",
        remake_dataset_if_not_found: bool = True,
        n_samples: int = 1000,
        diagonal_variation: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.default_model_dataset = default_model_dataset
        self.remake_dataset_if_not_found = remake_dataset_if_not_found
        self.n_samples = n_samples
        self.diagonal_variation = diagonal_variation
        self.rng: random.Random = random.Random(self.seed)

    def eval_sample(self, solver: Solver, sample: Any, rng: random.Random):
        previous_moves, next_filtered_moves = (
            sample["previous_moves"],
            sample["next_filtered_moves"],
        )

        def construct_controller(piece_id_to_instance: Dict[int, Piece]) -> BoardController:
            controller = BoardController(
                default_board_init,
                piece_id_to_instance,
                PIECE_STR_TO_ID,
                PIECE_ID_TO_STR,
                AlgebraicNotationParser(PIECE_STR_TO_ID, PIECE_ID_TO_STR),
            )
            for move in previous_moves:
                controller.update_board(move)
            return controller

        default_controller = construct_controller(PIECE_ID_TO_INSTANCE)
        variant_controller = construct_controller(VARIANT_PIECE_ID_TO_INSTANCE)

        # Get solver prediction. Ideally I wouldn't pass the legal_moves to the solvers, they
        # should figure them out themselves, but it's necessary for the random solver
        def get_solver_pred(
            task_description: str,
            controller: BoardController,
        ) -> SolverResult:
            task_state = TaskState(
                task_description,
                messages=construct_messages(previous_moves),
            )
            return solver(task_state, **{"max_tokens": 4})

        solver_result = get_solver_pred(TASK_DESCRIPTION, default_controller)
        solver_result_variant = get_solver_pred(TASK_DESCRIPTION_VARIANT, variant_controller)

        metrics = {
            "move": next_filtered_moves,
            "predicted_move": solver_result.output.strip() in next_filtered_moves,
            "predicted_move_in_variant": solver_result_variant.output.strip()
            in next_filtered_moves,
            "num_previous_moves": len(previous_moves),
            "previous_moves": previous_moves,
        }

        # Add violations to metrics
        metrics.update(
            self.get_violations(
                default_controller, solver_result.output, previous_moves, "standard"
            )
        )
        metrics.update(
            self.get_violations(
                variant_controller, solver_result_variant.output, previous_moves, "variant"
            )
        )

        evals.record.record_metrics(**metrics)

    def _run_impl(self, recorder: RecorderBase) -> dict[str, Union[float, int]]:
        if self.diagonal_variation:
            self.samples_jsonl = get_diagonal_dataset_path(
                registry_path=self._prefix_registry_path("")
            )
        else:
            self.samples_jsonl = get_dataset_path(
                solver=self._solver,
                registry_path=self._prefix_registry_path(""),
                remake_dataset_if_not_found=self.remake_dataset_if_not_found,
                default_model_dataset=self.default_model_dataset,
            )
        samples = self.get_samples()
        samples = self.rng.sample(samples, min(self.n_samples, len(samples)))

        self.eval_all_samples(recorder, samples)
        metrics = recorder.get_metrics()

        predicted_move_proportion = get_binary_avg(metrics, "predicted_move")
        predicted_move_in_variant_proportion = get_binary_avg(metrics, "predicted_move_in_variant")

        avg_num_previous_moves = sum([i["num_previous_moves"] for i in metrics]) / len(metrics)
        std_num_previous_moves = np.std([i["num_previous_moves"] for i in metrics])

        delta = predicted_move_in_variant_proportion - predicted_move_proportion
        variant_impact_factor = (delta / predicted_move_proportion) if predicted_move_proportion != 0 else 0

        results = {
            "variant_impact_factor": variant_impact_factor,
            "delta": delta,
            "predicted_move_proportion": predicted_move_proportion,
            "predicted_move_in_variant_proportion": predicted_move_in_variant_proportion,
            "avg_num_previous_moves": avg_num_previous_moves,
            "std_num_previous_moves": std_num_previous_moves,
        }

        # Add violations
        violation_keys = [i for i in metrics[0].keys() if "violation" in i]
        violation_results = {
            f"{name}_rate": get_binary_avg(metrics, name) for name in violation_keys
        }
        results.update(violation_results)

        return results

    def get_violations(
        self,
        controller: BoardController,
        solver_output: str,
        previous_moves: Sequence[str],
        variant_name: str,
    ) -> dict:
        solver_color = "W" if len(previous_moves) % 2 == 0 else "B"

        piece_moved_outside_board = False
        moving_invalid_piece = False
        piece_capturing_same_color = False

        violation_metrics = {}
        try:
            move = controller.notation_parser._str_to_move(
                solver_output, controller.board.board_state
            )

            piece_moved_outside_board = not move_within_board(move)
            moving_invalid_piece = not same_color_piece_at_move_start(
                controller.board.board_state, move, solver_color
            )
            piece_capturing_same_color = capturing_same_color(controller.board.board_state, move)
            incorrect_notation = False
        except (ValueError, KeyError):
            incorrect_notation = True

        violation = (
            piece_moved_outside_board
            or moving_invalid_piece
            or piece_capturing_same_color
            or incorrect_notation
        )
        violation_metrics = {
            f"{variant_name}_violation": violation,
            f"{variant_name}_violation_moved_outside_board": piece_moved_outside_board,
            f"{variant_name}_violation_moving_invalid_piece": moving_invalid_piece,
            f"{variant_name}_violation_capturing_same_color": piece_capturing_same_color,
            f"{variant_name}_violation_incorrect_notation": incorrect_notation,
        }
        return violation_metrics
