import base64
import io
import queue
from concurrent.futures import process as futures_process
from typing import Any, Dict, Optional

import librosa
import torch
import torch.distributed
import torch.multiprocessing as mp
import torch.nn.functional as F
import transformers

from evals.solvers.solver import Solver, SolverResult
from evals.task_state import TaskState

SAMPLE_RATE = 16000
MAX_BATCH_SIZE = 32


class FixieGPUSolver(Solver):
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
        model: str,
        num_gpus: int = torch.cuda.device_count(),
        extra_options: Optional[Dict[str, Any]] = None,
        postprocessors: list[str] = [],
        registry: Any = None,
    ):
        super().__init__(postprocessors=postprocessors, registry=registry)

        if extra_options is None:
            extra_options = {}

        # Set the start method for the entire script
        mp.set_start_method("spawn")

        rank_queue = mp.Queue()

        # First, we only start the primary process/GPU to let it download the model and avoid race-conditions
        rank_queue.put(0)

        if "max_new_tokens" not in extra_options:
            extra_options["max_new_tokens"] = 256

        self.executor = BatchedProcessPoolExecutor(
            max_workers=max(1, num_gpus),
            initializer=solver_initializer,
            initargs=(rank_queue, num_gpus, model, extra_options),
        )

    def copy(self):
        # The queue objects (in self.executor) must not be copied
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
            inputs["sampling_rate"] = SAMPLE_RATE

        inputs["turns"] = msgs

        # This is where the magic happens: we send the inputs to be processed by the model
        completion_output = self.executor.submit(solver_worker, inputs).result()

        solver_result = SolverResult(completion_output)

        return solver_result

    def __del__(self):
        if hasattr(self, "executor"):
            self.executor.shutdown()


class DataCollatorForSeq2SeqWithAudio(transformers.DataCollatorForSeq2Seq):
    def __call__(self, features, *args, **kwargs):
        audio_values = [f.pop("audio_values", None) for f in features]
        input_ids_lens = torch.LongTensor([f["input_ids"].shape[-1] for f in features])
        batch = super().__call__(features, *args, **kwargs)

        # Pad the last dimension of all audio_values to the same length, with 0s on the right.
        if audio_values and audio_values[0] is not None:
            max_len = max([x.shape[-1] for x in audio_values])
            batch["audio_values"] = torch.stack(
                [F.pad(x, (0, max_len - x.shape[-1])) for x in audio_values]
            )
            if self.tokenizer.padding_side == "left":
                displacement = batch["input_ids"].shape[-1] - input_ids_lens
                batch["audio_token_start_idx"] += displacement.to(
                    batch["audio_token_start_idx"].device
                )

        return batch


def solver_initializer(
    rank_queue: mp.Queue, world_size: int, model: str, extra_options: Dict[str, Any]
):
    """Initializes the pipeline and the underlying model on the specified GPU."""
    rank = rank_queue.get()

    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cpu")

    global pipe, collator

    pipe = transformers.pipeline(
        model=model,
        trust_remote_code=True,
        device=device,
        torch_dtype=torch.bfloat16,
        **extra_options,
    )
    pipe.tokenizer.padding_side = "left"

    collator = DataCollatorForSeq2SeqWithAudio(tokenizer=pipe.tokenizer)

    if rank == 0:
        # Let the other initializers start now that the download has finished
        for i in range(1, world_size):
            rank_queue.put(i)


def solver_worker(inputs: Dict[str, Any]):
    prepped = [pipe.preprocess(item) for item in inputs]
    prepped = [
        {k: v.to(pipe.model.device).squeeze(0) for k, v in sample.items()} for sample in prepped
    ]

    batch = collator(prepped)
    batch = {k: v.to(pipe.model.device) for k, v in batch.items() if v is not None}

    with torch.inference_mode():
        terminators = [pipe.tokenizer.eos_token_id]
        if "<|eot_id|>" in pipe.tokenizer.added_tokens_encoder:
            terminators.append(pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>"))

        input_len = batch["input_ids"].shape[1]

        outputs = pipe.model.generate(
            **batch,
            eos_token_id=terminators,
            **pipe._forward_params,
        )
        out_texts = [
            pipe.tokenizer.decode(o[input_len:], skip_special_tokens=True) for o in outputs
        ]
        return out_texts


class _BatchedExectuorManagerThread(futures_process._ExecutorManagerThread):
    """
    This is a version of the _ExecutorManagerThread class that allows for
    batching the inputs. This is necessary for efficiently generating completions.

    NOTE: This class assumes that all submitted work items use the same function and that
    they only have a single argument and no keyword arguments. Extending this is left for future work.
    """

    def add_call_item_to_queue(self):
        # Fills call_queue with _WorkItems from pending_work_items.
        # This function never blocks.
        while True:
            if self.call_queue.full():
                return

            work_ids = []
            while len(work_ids) < MAX_BATCH_SIZE:
                try:
                    work_id = self.work_ids_queue.get(block=False)
                    work_ids.append(work_id)
                except queue.Empty:
                    break

            if not work_ids:
                return

            work_items = [self.pending_work_items[work_id] for work_id in work_ids]
            is_not_cancelled = [item.future.set_running_or_notify_cancel() for item in work_items]
            work_items = [
                item
                for item, is_not_cancelled in zip(work_items, is_not_cancelled)
                if is_not_cancelled
            ]

            if work_items:
                self.call_queue.put(
                    futures_process._CallItem(
                        work_ids,
                        work_items[0].fn,
                        ([work_item.args[0] for work_item in work_items],),
                        {},
                    ),
                    block=True,
                )

    def process_result_item(self, result_item):
        # Process the received a result_item. This can be either the PID of a
        # worker that exited gracefully or a _ResultItem

        if isinstance(result_item, int):
            # Clean shutdown of a worker using its PID
            # (avoids marking the executor broken)
            assert self.is_shutting_down()
            p = self.processes.pop(result_item)
            p.join()
            if not self.processes:
                self.join_executor_internals()
                return
        else:
            # Everything above this line is the same as the parent class
            # Received a _ResultItem so mark the future as completed.
            result_work_ids = result_item.work_id
            work_items = [self.pending_work_items.pop(work_id, None) for work_id in result_work_ids]
            # work_item can be None if another process terminated (see above)
            for i, work_item in enumerate(work_items):
                if work_item is not None:
                    if result_item.exception:
                        work_item.future.set_exception(result_item.exception)
                    else:
                        work_item.future.set_result(result_item.result[i])


class BatchedProcessPoolExecutor(futures_process.ProcessPoolExecutor):
    """
    This is a ProcessPoolExecutor that uses a custom _ExecutorManagerThread to allow batching of inputs.
    """

    def _start_executor_manager_thread(self):
        # Everything in this function is the same as the parent class, except for:
        # 1. _ExectuorManagerThread  ->  _BatchedExectuorManagerThread class
        # 2. _threads_wakeups  ->  futures_process._threads_wakeups
        if self._executor_manager_thread is None:
            # Start the processes so that their sentinels are known.
            if not self._safe_to_dynamically_spawn_children:  # ie, using fork.
                self._launch_processes()
            self._executor_manager_thread = _BatchedExectuorManagerThread(self)
            self._executor_manager_thread.start()
            futures_process._threads_wakeups[
                self._executor_manager_thread
            ] = self._executor_manager_thread_wakeup
