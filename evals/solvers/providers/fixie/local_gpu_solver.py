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

SAMPLE_RATE = 16000


def solver_initializer(rank_queue: mp.Queue, world_size: int, model: str):
    """Initializes the pipeline and the underlying model on the specified GPU."""
    print("is dist inited?", torch.distributed.is_initialized())
    rank = rank_queue.get()

    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    else:
        raise ValueError("Only GPU mode is supported by LocalGPUSolver, but no GPUs were found")

    global pipe

    with run_on_master_first(is_master=rank == 0):
        pipe = transformers.pipeline(model=model, trust_remote_code=True, device=device)


def solver_worker(inputs: Dict[str, Any]):
    return pipe(inputs, max_new_tokens=64)


class LocalGPUSolver(Solver):
    """
    A solver class for the locally running a model on multiple GPUs using a ProcessPoolExecutor.
    The model is employed using a Hugging Face Transformers pipeline. We assume that the pipeline
    is an UltravoxPipeline class and the inputs are text/audio messages.

    Completion function parameters:
    - model: str - The model name/path to use for the pipeline
    - num_gpus: int - The number of GPUs to use for parallel processing (default: all available GPUs)
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
        completion_fn_options.get("max_num_tokens", 64)

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
            # This means the last message is an audio message
            parts = msgs[-1]["content"]
            parts_str = [x["text"] if x["type"] == "text" else "<|audio|>" for x in parts]
            # Concatenate all text parts into a single string
            msgs[-1]["content"] = "".join(parts_str)
            data_parts = [x["image_url"] for x in parts if x["type"] == "image_url"]
            assert len(data_parts) == 1
            # Extract the audio data from the last message
            audio_data = data_parts[0]["url"].split(",")[1]
            audio_stream = io.BytesIO(base64.b64decode(audio_data))

            # Read the audio data using soundfile and enforce the expected sample rate
            inputs["audio"] = librosa.load(audio_stream, sr=SAMPLE_RATE)[0]

        inputs["turns"] = msgs

        completion_output = self.executor.submit(solver_worker, inputs).result()

        solver_result = SolverResult(completion_output)

        return solver_result

    def __del__(self):
        self.executor.shutdown()


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
