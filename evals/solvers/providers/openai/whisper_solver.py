# from evals.solvers.providers.openai.openai_solver import OpenAISolver
import base64
import io
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Optional

import librosa
import openai
import torch
import torch.distributed
import torch.multiprocessing as mp
import transformers

from evals.solvers.solver import NestedSolver, Solver, SolverResult, SolverSpec
from evals.task_state import Message, TaskState


class WhisperCascadedSolver(NestedSolver):
    """
    This solver rewrites any supplied messages with audio content to only have text content,
    by transcribing the audio using Whisper (Large).
    """

    def __init__(
        self,
        solver: SolverSpec,
        postprocessors: list[str] = [],
        registry: Any = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        super().__init__(postprocessors=postprocessors, solver=solver)
        # Initializing this in the ctor leads to strange pickle errors.
        self.client: Optional[openai.OpenAI] = None
        self.base_url = base_url
        self.api_key = api_key

    @property
    def solver(self) -> Solver:
        return self.get_solver("solver")

    def _solve(
        self,
        task_state: TaskState,
        **kwargs,
    ) -> SolverResult:
        task_state.messages = self._process_msgs(task_state.messages)
        return self.solver(task_state=task_state, **kwargs)

    @property
    def name(self) -> str:
        return f"{self.solver.name}_whisper"

    def _process_msgs(self, raw_msgs: list[Message]) -> list[Message]:
        return [self._process_message(msg) for msg in raw_msgs]

    def _process_message(self, msg: Message) -> Message:
        if msg.role == "user" and isinstance(msg.content, list):
            parts = [self._process_part(part) for part in msg.content]  # type: ignore
            if all([part["type"] == "text" for part in parts]):
                msg.content = "\n".join([part["text"] for part in parts])
        return msg

    def _process_part(self, part: dict[str, Any]) -> dict[str, Any]:
        if part["type"] == "audio_url":
            url = part["audio_url"]["url"]
            if url.startswith("data:audio/x-wav;base64,"):
                wav_b64 = url.split(",")[1]
                wav_bytes = base64.b64decode(wav_b64)
                text = self._transcribe(wav_bytes)
                part = {
                    "type": "text",
                    "text": text,
                }

        return part

    def _transcribe(self, wav_bytes: bytes) -> str:
        if not self.client:
            self.client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key)
        file = io.BytesIO(wav_bytes)
        file.name = "test.wav"
        return self.client.audio.transcriptions.create(model="whisper-1", file=file).text


# TODO: batched version
class WhisperCascadedGPUSolver(WhisperCascadedSolver):
    def __init__(self, model: str, **kwargs):
        super().__init__(**kwargs)

        # Set the start method for the entire script
        mp.set_start_method("spawn")

        rank_queue = mp.Queue()

        # Only start the primary to let it download the model first
        rank_queue.put(0)

        num_gpus = torch.cuda.device_count()

        self.executor = ProcessPoolExecutor(
            max_workers=max(1, num_gpus),
            initializer=solver_initializer,
            initargs=(rank_queue, num_gpus, model),
        )

    def copy(self):
        return self

    def _transcribe(self, wav_bytes: bytes) -> str:
        file = io.BytesIO(wav_bytes)
        file.name = "test.wav"

        audio = librosa.load(file, sr=16000)[0]

        return self.executor.submit(solver_worker, audio).result()

    def __del__(self):
        if hasattr(self, "executor"):
            self.executor.shutdown()


def solver_initializer(rank_queue: mp.Queue, world_size: int, model: str):
    """Initializes the pipeline and the underlying model on the specified GPU."""
    rank = rank_queue.get()

    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cpu")

    global pipe

    pipe = transformers.pipeline(
        "automatic-speech-recognition",
        model=model,
        chunk_length_s=30,
        device=device,
    )

    if rank == 0:
        # Let the other initializers start now that the download has finished
        for i in range(1, world_size):
            rank_queue.put(i)


def solver_worker(audio):
    with torch.inference_mode():
        return pipe(audio)["text"]
