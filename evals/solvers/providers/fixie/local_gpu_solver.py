from typing import Any, Dict

import torch
import torch.multiprocessing as mp
import transformers

from evals.solvers.solver import Solver, SolverResult
from evals.task_state import TaskState


def solver_worker(rank: int, queue: mp.Queue, model: str):
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)

    pipe = transformers.pipeline(model=model, trust_remote_code=True, device=device)

    inputs: Dict[str, Any]
    result_queue: mp.Queue

    while True:
        inputs, result_queue = queue.get()
        completion = pipe(inputs)
        result_queue.put(completion)


class LocalGPUSolver(Solver):
    """
    TODO
    """

    def __init__(self, completion_fn_options: Dict[str, Any] = {}, **kwargs):
        super().__init__(kwargs)
        self.completion_fn_options = completion_fn_options

        num_gpus = completion_fn_options.get("num_gpus", torch.cuda.device_count())

        if "model" not in completion_fn_options:
            raise ValueError("LocalGPUSolver requires a model to be specified.")

        self.queue = mp.Queue()

        # TODO: handle num_gpus=0 differently
        mp.spawn(solver_worker, args=(self.queue, completion_fn_options["model"]), nprocs=num_gpus)

    def _solve(self, task_state: TaskState, **kwargs) -> SolverResult:
        msgs = [
            {"role": "system", "content": task_state.task_description},
        ] + [msg.to_dict() for msg in task_state.messages]

        inputs = {"turns": msgs}

        # create a way for the worker to return the result
        result_queue = mp.Queue()
        # send the inputs to the worker
        self.queue.put((inputs, result_queue))
        # get the result back from the worker
        completion_output = result_queue.get()

        solver_result = SolverResult(completion_output)

        return solver_result
