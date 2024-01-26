import logging
import time
from threading import Lock
from typing import Any, Dict, Optional

import openai
from openai.types.beta import Assistant
from openai.types.beta.thread import Thread
from openai.types.beta.threads.run import Run

from evals.record import record_sampling
from evals.registry import client
from evals.solvers.solver import Solver, SolverResult
from evals.task_state import Message, TaskState

FILE_CACHE_LOCK = Lock()
FILE_CACHE = {}  # File cache can be reused across solvers


class OpenAIAssistantsSolver(Solver):
    """
    A solver that uses the OpenAI Assistants API to solve tasks. If you are new to Assistants, please
    start by reading the overview to understand how Assistants work:
    https://platform.openai.com/docs/assistants/overview

    Features:
    - Works with any tools (e.g. `code-interpreter`, `retrieval`) that are supported by Assistants.
      To use a tool, add it to the `tools` argument when instantiating the solver.
    - Supports file reading via the `code-interpreter` and `retrieval` tools. To use a file, add it
      to the `file_paths` argument when instantiating the solver (the file will be available to all
      threads). To use a file in a specific thread, add it to the `files` argument in the
      `TaskState.current_state` object.

    Special notes:
    - IMPORTANT: The Assistants API is priced differently than the Chat and Completion APIs. Please
      familiarize yourself with https://openai.com/pricing to avoid unexpected charges.
    - Each instantiation of the OpenAIAssistantsSolver class creates a new Assistant and Thread.
    - `solver.copy()` will create a new Thread but reuse the same Assistant.
    - The Assistant is stateful, so it is not possible to modify the history of messages, and
      the Solver assumes that new messages are sent after the last Assistant message.
    - The Assistants API is still in beta, so some features are not yet stable (e.g. assistants
      using the retrieval tool need to be reminded in-chat to read the file).
    - This solver does not provide support for none-text content in messages yet (e.g. images).
    """

    def __init__(
        self,
        model: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tools: list[Dict[str, Any]] = [],
        file_paths: list[str] = [],
        assistant: Optional[Assistant] = None,
        thread: Optional[Thread] = client.beta.threads.create(),
        registry: Any = None,
    ):
        self.model = model
        self.thread = thread
        self.tools = tools
        self.all_uploaded_files = []
        if not assistant:
            file_ids = self._create_files(file_paths)
            self.assistant = client.beta.assistants.create(
                model=model,
                name=name,
                description=description,
                tools=tools,
                file_ids=file_ids,  # Files attached here are available to all threads.
            )
        else:
            # This is a special init case for copying the solver - see `OpenAIAssistantsSolver.copy()`
            assert (
                not name and not description and not tools and not file_paths
            ), "Cannot specify `name`, `description`, `tools`, or `file_paths` when copying a solver."
            self.assistant = assistant

    def _solve(
        self,
        task_state: TaskState,
        **kwargs,
    ) -> SolverResult:
        """
        ARGS
        ====
        `task_state`: A `TaskState` object that contains the task description and the input.
        `kwargs`: Other arguments passed to the solver.

        RETURNS
        =======
        The result of the solver.
        """

        # Upload thread-specific files
        thread_file_ids = []
        if task_state.current_state is not None and "files" in task_state.current_state:
            thread_file_ids = self._create_files(task_state.current_state["files"])

        # We only send new messages to the Assistant since the Assistant is stateful.
        # This assumes that any new messages happen after the last Assistant message.
        last_assistant_msg_idx = self._get_last_assistant_message_idx(task_state.messages)
        new_msgs_start_idx = last_assistant_msg_idx + 1 if last_assistant_msg_idx is not None else 0

        # Add new messages to Thread
        last_msg_sent = None
        for idx, message in enumerate(task_state.messages[new_msgs_start_idx:]):
            user_message = self._convert_to_user_message(message)  # API only allows "user" messages
            last_msg_sent = client.beta.threads.messages.create(
                thread_id=self.thread.id,
                role=user_message.role,
                content=user_message.content,
                file_ids=thread_file_ids
                if idx == 0
                else [],  # Attach files to first new message only
            )

        # Run Assistant on the Thread
        run = client.beta.threads.runs.create(
            assistant_id=self.assistant.id,
            thread_id=self.thread.id,
            instructions=task_state.task_description,  # Apply task description as `instructions`
        )
        run = self._wait_on_run(run, self.thread)
        if run.status != "completed":
            error_msg = f"Assistants API Run failed with status {run.status}. More details: {run}"
            raise RuntimeError(error_msg)

        # Get Assistant response(s)
        messages = client.beta.threads.messages.list(
            thread_id=self.thread.id,
            order="asc",
            after=last_msg_sent.id if last_msg_sent else None,
        )

        contents = []
        for message in messages:
            for content in message.content:
                if content.type == "text":
                    contents.append(content.text.value)
                    # TODO: Handle content.text.annotations ?
                elif content.type == "image_file":
                    contents.append("{Assistant sent an image}")
                    logging.warning("Assistant sent an image, but this is not yet supported.")
                else:
                    raise NotImplementedError(f"Content type {content.type} not supported.")
        output_text = "\n".join(contents)

        # TODO: The Assistant also reports Run Steps which detail logs for tool use
        # https://platform.openai.com/docs/api-reference/runs/listRunSteps

        record_sampling(
            prompt=task_state.messages,
            sampled=[output_text],
            model=self.model,
            tools=self.tools,
            assistant=self.assistant.id,
            thread=self.thread.id,
            uploaded_files=self.all_uploaded_files,
        )
        return SolverResult(
            output=output_text,
        )

    def copy(self):
        # Assistants don't support copying; each sample uses the same Assistant but interacts with
        # a new Thread.

        # Return the a solver that uses the same Assistant, but give it a new Thread
        solver_copy = OpenAIAssistantsSolver(
            model=self.model,
            assistant=self.assistant,
            thread=client.beta.threads.create(),
        )
        return solver_copy

    def _create_file(self, file_path: str) -> str:
        with FILE_CACHE_LOCK:
            # If file is already uploaded, just reuse the same file
            if file_path in FILE_CACHE:
                return FILE_CACHE[file_path]
            try:
                file = client.files.create(
                    file=open(file_path, "rb"),
                    purpose="assistants",
                )
                FILE_CACHE[file_path] = file.id
                self.all_uploaded_files.append((file_path, file.id))
            except openai.BadRequestError as e:
                if "Invalid file format." in e.message:
                    logging.warning(f"{file_path} rejected due to invalid file format, skipping.")
                    return None
                else:
                    raise e
        return file.id

    def _create_files(self, file_paths: list[str]) -> list[str]:
        file_ids = []
        for file_path in file_paths:
            file_id = self._create_file(file_path)
            if file_id is not None:
                file_ids.append(file_id)
        return file_ids

    def _get_last_assistant_message_idx(self, messages: list[Message]) -> Optional[int]:
        last_idx = None
        for i, message in enumerate(messages):
            if message.role == "assistant":
                last_idx = i
        return last_idx

    def _convert_to_user_message(self, message: Message) -> Message:
        """
        Assistants API only allows "user" messages, so all other role (e.g. "system") must be rendered
        into "user" messages.
        """
        if message.role != "user":
            message.content = f"[{message.role}] {message.content}"
            message.role = "user"
        return message

    def _wait_on_run(self, run: Run, thread: Thread) -> Run:
        """
        Wait for run to finish. (End state may be "completed", "expired", "failed" or "cancelled".)
        Function borrowed from: https://cookbook.openai.com/examples/assistants_api_overview_python
        """
        while run.status == "queued" or run.status == "in_progress":
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id,
            )
            time.sleep(0.5)
        return run

    @property
    def name(self) -> str:
        return f"OpenaiAssistantsSolver_{self.name}_{self.model}"
