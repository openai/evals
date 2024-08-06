import base64
import contextlib
import io
from typing import Any, Dict

import soundfile as sf
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


def solver_worker(rank: int, world_size: int, queue: mp.Queue, model: str):
    print("is dist inited?", torch.distributed.is_initialized())
    print("queue", queue)

    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    else:
        raise ValueError("Only GPU mode is supported by LocalGPUSolver, but no GPUs were found")

    with run_on_master_first(is_master=rank == 0):
        pipe = transformers.pipeline(model=model, trust_remote_code=True, device=device)

    inputs: Dict[str, Any]
    result_queue: mp.Queue

    while True:
        inputs, result_queue = queue.get()
        print("got intput", inputs)
        if inputs is None:
            # indicates the end of inputs / kill process
            return

        pipe(inputs)
        # result_queue.put(completion)


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

        self.queue = mp.Queue()

        # TODO: handle num_gpus=0 differently
        self.process_context = mp.spawn(
            solver_worker,
            args=(num_gpus, self.queue, completion_fn_options["model"]),
            nprocs=num_gpus,
            join=False,
            daemon=True,
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
            audio_data = base64.b64decode(data_parts[0])
            audio_stream = io.BytesIO(audio_data)

            # Read the audio data using soundfile
            inputs["audio"] = sf.read(audio_stream, samplerate=16000)[0]

        inputs["turns"] = msgs

        # create a way for the worker to return the result
        # result_queue = mp.Queue()
        # send the inputs to the worker
        self.queue.put((inputs, None))
        # get the result back from the worker
        # completion_output = None.get()
        completion_output = "True"

        solver_result = SolverResult(completion_output)

        return solver_result

    def __del__(self):
        if getattr(self, "process_context", None):
            for _ in self.process_context.processes:
                self.queue.put((None, None))

            self.process_context.join()
