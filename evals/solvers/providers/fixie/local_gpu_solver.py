import base64
import contextlib
import io
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict

import librosa
import torch
import torch.distributed
import torch.multiprocessing as mp
import transformers

from evals.solvers.solver import Solver, SolverResult
from evals.task_state import TaskState


@contextlib.contextmanager
def run_on_master_first(is_master: bool):
    """
    If using DDP, allows the master process to run the enclosed code first.
    This is useful when only one process should download a model or other resources first to avoid race conditions.
    """
    if is_master:
        yield
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    else:
        # All other processes wait for the master to download the model first
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        yield


def solver_initializer(rank_queue: mp.Queue, world_size: int, model: str):
    print("is dist inited?", torch.distributed.is_initialized())
    # torch.distributed.recv
    # print("queue", queue)
    rank = rank_queue.get()

    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    else:
        raise ValueError("Only GPU mode is supported by LocalGPUSolver, but no GPUs were found")

    global pipe

    with run_on_master_first(is_master=rank == 0):
        pipe = transformers.pipeline(model=model, trust_remote_code=True, device=device)


def solver_worker(inputs: Dict[str, Any]):
    global pipe
    return pipe(inputs, max_new_tokens=64)


class LocalGPUSolver(Solver):
    """
    TODO
    """

    def __init__(
        self,
        completion_fn_options: Dict[str, Any] = {},
        postprocessors: list[str] = [],
        registry: Any = None,
    ):
        super().__init__(postprocessors=postprocessors)
        self.completion_fn_options = completion_fn_options

        num_gpus = completion_fn_options.get("num_gpus", torch.cuda.device_count())

        if "model" not in completion_fn_options:
            raise ValueError("LocalGPUSolver requires a model to be specified.")

        # Set the start method for the entire script
        mp.set_start_method("spawn")

        rank_queue = mp.Queue()
        for i in range(num_gpus):
            rank_queue.put(i)

        # # TODO: handle num_gpus=0 differently
        self.executor = ProcessPoolExecutor(
            max_workers=num_gpus,
            initializer=solver_initializer,
            initargs=(rank_queue, num_gpus, completion_fn_options["model"]),
        )

    def copy(self):
        # FIXME: ignoring copy due to MP error
        return self

    def _solve(self, task_state: TaskState, **kwargs) -> SolverResult:
        inputs = {}
        msgs = [
            {"role": "system", "content": task_state.task_description},
        ] + [msg.to_dict() for msg in task_state.messages]

        if not isinstance(msgs[-1]["content"], str):
            parts = msgs[-1]["content"]
            parts_str = [x["text"] if x["type"] == "text" else "<|audio|>" for x in parts]
            msgs[-1]["content"] = "".join(parts_str)
            data_parts = [x["image_url"] for x in parts if x["type"] == "image_url"]
            assert len(data_parts) == 1
            audio_data = data_parts[0]["url"].split(",")[1]
            audio_data = base64.b64decode(audio_data)
            audio_stream = io.BytesIO(audio_data)

            # Read the audio data using soundfile
            inputs["audio"] = librosa.load(audio_stream, sr=16000)[0]

        inputs["turns"] = msgs

        completion_output = self.executor.submit(solver_worker, inputs).result()

        solver_result = SolverResult(completion_output)

        return solver_result

    def __del__(self):
        self.executor.shutdown()
