import json
import logging
import os
from multiprocessing.pool import ThreadPool
from typing import Sequence

import chess
from tqdm import tqdm

from evals.elsuite.cant_do_that_anymore.chess.board import BoardController
from evals.elsuite.cant_do_that_anymore.chess.board_test import default_board_init
from evals.elsuite.cant_do_that_anymore.chess.move_variants import (
    PIECE_ID_TO_INSTANCE,
    PIECE_ID_TO_STR,
    PIECE_STR_TO_ID,
    VARIANT_PIECE_ID_TO_INSTANCE,
)
from evals.elsuite.cant_do_that_anymore.chess.notation import AlgebraicNotationParser
from evals.elsuite.cant_do_that_anymore.defaults import TASK_DESCRIPTION
from evals.record import DummyRecorder, RecorderBase
from evals.solvers.solver import DummySolver, Solver
from evals.task_state import Message, TaskState

logger = logging.getLogger(__name__)


def construct_messages(previous_moves: Sequence[str]) -> Sequence[Message]:
    """
    Creates list of Message's containing the previous chess moves. The last
    Message is always from the "user"
    """
    solver_is_white = len(previous_moves) % 2 == 0
    messages = []
    current_player = "assistant" if solver_is_white else "user"
    for move in previous_moves:
        messages.append(Message(current_player, move))
        # toggle current player
        current_player = "assistant" if current_player == "user" else "user"

    return messages


def dump_sequence_to_jsonl(data: Sequence[dict], path: str):
    with open(path, "w+") as f:
        for example in data:
            example = json.dumps(example)
            f.write(f"{example}\n")


def load_sequence_from_jsonl(path: str) -> Sequence[dict]:
    data = []
    with open(path, "r") as f:
        for line in f:
            line = json.loads(line)
            data.append(line)

    return data


def initialise_boards() -> tuple[BoardController, BoardController, chess.Board]:
    """
    Initialises local chess framework, and framework from
    python-chess library
    """
    default_controller = BoardController(
        default_board_init,
        PIECE_ID_TO_INSTANCE,
        PIECE_STR_TO_ID,
        PIECE_ID_TO_STR,
        AlgebraicNotationParser(PIECE_STR_TO_ID, PIECE_ID_TO_STR),
    )
    variant_controller = BoardController(
        default_board_init,
        VARIANT_PIECE_ID_TO_INSTANCE,
        PIECE_STR_TO_ID,
        PIECE_ID_TO_STR,
        AlgebraicNotationParser(PIECE_STR_TO_ID, PIECE_ID_TO_STR),
    )
    their_controller = chess.Board()

    return default_controller, variant_controller, their_controller


def assert_boards_consistent(
    controller: BoardController, their_controller: chess.Board, player_id: str
):
    """
    Checks both boards have consistent states by ensuring both have same set of legal moves
    """
    our_legal_moves = sorted(controller.get_player_legal_moves(player_id))
    their_legal_moves = sorted([str(i) for i in their_controller.legal_moves])
    if our_legal_moves != their_legal_moves:
        our_additional_moves = list(set(our_legal_moves) - set(their_legal_moves))
        their_additional_moves = list(set(their_legal_moves) - set(our_legal_moves))
        assert False, f"""
                Inconsistent legal moves between the boards!
                Our legal moves: {our_legal_moves},
                Their legal moves: {their_legal_moves},
                Moves we had they didnt: {our_additional_moves},
                Moves they had we didn't: {their_additional_moves},
                Board state:\n{controller.board.board_state}
            """


def does_solver_predict_move(
    solver: Solver,
    recorder: RecorderBase,
    task_description: str,
    special_move: str,
    previous_moves: Sequence[str],
):
    task_state = TaskState(
        task_description,
        construct_messages(previous_moves),
    )

    with recorder.as_default_recorder(-1):
        solver_result = solver(task_state, **{"max_tokens": 4})
    pred_str = solver_result.output.strip()

    if pred_str == special_move:
        return True

    return False


def process_example(work_input: dict):
    solver, recorder, example, task_description = (
        work_input["solver"],
        work_input["recorder"],
        work_input["example"],
        work_input["task_description"],
    )
    special_move, previous_moves = example["special_move"], example["previous_moves"]

    predicts_move = does_solver_predict_move(
        solver,
        recorder,
        task_description,
        special_move,
        previous_moves,
    )
    return predicts_move, example


def get_solver_predictions(
    solver: Solver,
    recorder: RecorderBase,
    special_moves_dataset: Sequence[dict],
    n_threads: int,
    task_description: str,
) -> Sequence[dict]:
    """
    Filter to find all special moves that the solver would have predicted under the normal
    rules of chess with temp=0, then dump this dataset
    """
    solver_moves_dataset = []
    work_items = [
        {
            "solver": solver,
            "recorder": recorder,
            "example": example,
            "task_description": task_description,
        }
        for example in special_moves_dataset
    ]

    t_bar = tqdm(total=len(special_moves_dataset))
    with ThreadPool(n_threads) as pool:
        iter = pool.imap_unordered(process_example, work_items)

        for result in (t_bar := tqdm(iter, total=len(work_items))):
            predicts_move, example = result
            if predicts_move:
                solver_moves_dataset.append(example)
            t_bar.set_description(f"Dataset size: {len(solver_moves_dataset)}")

    return solver_moves_dataset


def get_dataset_path(
    solver: Solver,
    registry_path: str,
    remake_dataset_if_not_found: bool,
    default_model_dataset: str,
) -> str:
    """
    This dataset requires each evaluated model to have its own dataset. We get the exact
    model being exaluated, check if a dataset exists for it, if not we generate one
    """
    recorder = DummyRecorder(None)
    with recorder.as_default_recorder("x"):
        solver_version = solver.model_version

    # If nested solver, convert returned dict to str
    if isinstance(solver_version, dict):
        solver_version = json.dumps(solver_version)

    all_datasets_path = os.path.join(registry_path, "cant_do_that_anymore")

    # Check if dataset exists
    solver_dataset_path = os.path.join(all_datasets_path, f"{solver_version}_dataset.jsonl")
    if os.path.exists(solver_dataset_path):
        return solver_dataset_path

    # Remake, or load default
    if isinstance(solver, DummySolver):
        return f"cant_do_that_anymore/{default_model_dataset}_dataset.jsonl"
    elif remake_dataset_if_not_found:
        logger.warning(
            f"Generating dataset for {solver_version}! Ideally the solver should be using temperature=0 when creating the dataset, "
            "otherwise generated dataset will be of a slightly different distribution"
        )
        create_dataset(solver, recorder, solver_dataset_path, all_datasets_path)
        return solver_dataset_path
    else:
        logger.warning(
            f"Dataset for {solver_version} wasn't found! Using the dataset for {default_model_dataset} instead."
        )
        return f"cant_do_that_anymore/{default_model_dataset}_dataset.jsonl"


def create_dataset(
    solver: Solver, recorder: RecorderBase, solver_dataset_path: str, all_datasets_path: str
):
    threads = int(os.environ.get("EVALS_THREADS", "10"))

    special_moves_dataset = load_sequence_from_jsonl(
        os.path.join(all_datasets_path, "special_moves_dataset.jsonl")
    )
    solver_moves_dataset = get_solver_predictions(
        solver,
        recorder,
        special_moves_dataset,
        n_threads=threads,
        task_description=TASK_DESCRIPTION,
    )
    dump_sequence_to_jsonl(solver_moves_dataset, solver_dataset_path)


def get_diagonal_dataset_path(
    registry_path: str,
) -> str:
    return os.path.join(registry_path, "cant_do_that_anymore/diagonal_moves_dataset.jsonl")


def get_binary_avg(metrics: dict, key: str) -> float:
    positive_examples = [i for i in metrics if i[key]]
    avg = len(positive_examples) / len(metrics)
    return avg
