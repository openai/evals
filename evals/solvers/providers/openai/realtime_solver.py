import asyncio
import os
from typing import Any, Dict, Optional, Union

import websockets

from evals.solvers.solver import Solver, SolverResult
from evals.task_state import TaskState


class RealtimeSolver(Solver):
    """ """

    def __init__(
        self,
        completion_fn_options: Dict[str, Any],
        postprocessors: list[str] = [],
        registry: Any = None,
    ):
        super().__init__(postprocessors=postprocessors)
        if "model" not in completion_fn_options:
            raise ValueError("OpenAISolver requires a model to be specified.")
        self.completion_fn_options = completion_fn_options

    @property
    def model(self) -> str:
        """
        Get model name from completion function, e.g. "gpt-3.5-turbo"
        This may not always include the full model version, e.g. "gpt-3.5-turbo-0613"
        so use `self.model_version` if you need the exact snapshot.
        """
        return self.completion_fn_options["model"]

    def name(self) -> str:
        return self.model

    @property
    def model_version(self) -> Union[str, dict]:
        return {}

    @property
    def _api_base(self) -> Optional[str]:
        """The base URL for the API"""
        return "wss://api.openai.com/v1/realtime"

    @property
    def _api_key(self) -> Optional[str]:
        """The API key to use for the API"""
        return os.getenv("OPENAI_API_KEY")

    def _solve(self, task_state: TaskState, **kwargs) -> SolverResult:
        raw_msgs = [
            {"role": "system", "content": task_state.task_description},
        ] + [msg.to_dict() for msg in task_state.messages]
        completion_result = asyncio.run(self._foo(raw_msgs))
        completion_output = completion_result.get_completions()[0]
        solver_result = SolverResult(completion_output, raw_completion_result=completion_result)
        return solver_result

    async def _foo(self, messages):
        url = f"{self._api_base}?model={self.model}"
        headers = {"Authorization": f"Bearer {self._api_key}", "OpenAI-Beta": "realtime=v1"}
        print(url, headers)
        async with websockets.connect(url, extra_headers=headers) as websocket:
            # session_event = {"type": "session.update", "model": "gpt-4o-realtime-preview-2024-10-01"} ?
            # await websocket.send(json.dumps(session_event))

            # Send each message in the list
            for message in messages:
                message["role"]
                message["content"]
                # await websocket.send(json.dumps(event))
            # await websocket.send(create_response)

            response = await websocket.recv()
            await websocket.close()
            return response
