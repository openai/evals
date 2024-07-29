# from evals.solvers.providers.openai.openai_solver import OpenAISolver
from evals.solvers.solver import NestedSolver, Solver, SolverResult, SolverSpec
from evals.task_state import Message, TaskState

from typing import Any, Optional
import openai
import base64
import io


class WhisperSolver(NestedSolver):
    """
    This solver rewrites any supplied messages with audio content to only have text content,
    by transcribing the audio using Whisper.
    """

    def __init__(
        self,
        solver: SolverSpec,
        postprocessors: list[str] = [],
        registry: Any = None,
    ):
        super().__init__(postprocessors=postprocessors, solver=solver)
        self.client: Optional[openai.OpenAI] = None

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

    def _process_msgs(self, raw_msgs: list[dict[str, str]]) -> list[dict[str, str]]:
        return [self._process_message(msg) for msg in raw_msgs]

    def _process_message(self, msg: dict[str, str]) -> dict[str, str]:
        if msg.role == "user" and isinstance(msg.content, list):
            parts = [self._process_part(part) for part in msg.content]
            if all([part["type"] == "text" for part in parts]):
                msg.content = "\n".join([part["text"] for part in parts])
        return msg

    def _process_part(self, part: dict[str, str]) -> dict[str, str]:
        if part["type"] == "image_url":
            url = part["image_url"]["url"]
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
            self.client = openai.OpenAI()
        file = io.BytesIO(wav_bytes)
        file.name = "test.wav"
        return self.client.audio.transcriptions.create(model="whisper-1", file=file).text
