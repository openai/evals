import copy
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from evals.elsuite.bugged_tools.tools import Tool, ToolTaskState
from evals.solvers.solver import Solver, SolverResult
from evals.task_state import Message, TaskState

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    tool_name: str
    input: str
    output: Any


@dataclass
class ParsedSolverResult:
    tool_calls: list[ToolCall]
    final_answer: Optional[str]


@dataclass
class RunnerResult:
    final_task_state: ToolTaskState
    final_solver_result: SolverResult
    metrics: dict


class Runner:
    def __init__(
        self,
        solver: Solver,
        sample: Any,
        name_to_tool: dict,
        max_turns: int,
        default_task_description: str,
        default_reminder_message: str,
    ):
        self.solver = solver
        self.sample = sample
        self.name_to_tool = name_to_tool
        self.max_turns = max_turns
        self.default_task_description = default_task_description
        self.default_reminder_message = default_reminder_message

    def run(self) -> RunnerResult:
        # Prepare initial task state
        tools = self.name_to_tool.values()
        tool_names_and_descriptions = self._get_tool_names_and_descriptions(tools)
        task_description = self.default_task_description.format(
            tool_names_and_descriptions=tool_names_and_descriptions
        )
        task_message = self.sample["task"]
        messages = [
            Message(role="user", content=task_message),
        ]
        task_state = TaskState(
            task_description=task_description,
            messages=messages,
            current_state=None,
        )

        # Loops until solver completes task or hits turn limit
        turn = 0
        final_answer = None
        while turn < self.max_turns:
            # Get result from solver
            solver_result = self.solver(task_state)
            parsed_solver_result = self._parse_solver_result(solver_result)
            final_answer = parsed_solver_result.final_answer

            # If solver failed to call tool or give final answer, prompt them to try again
            if parsed_solver_result.tool_calls == [] and final_answer is None:
                content = self.default_reminder_message
                task_state = self._add_eval_message(task_state, solver_result, content=content)
                turn += 1
                continue

            if final_answer is not None:
                return self._finish_run(task_state, solver_result, final_answer, turn)

            # Run tools. If solver gave tool incorrect input, prompt them to try again.
            assert parsed_solver_result.tool_calls != []
            tool_outputs = [self._run_tool_call(i) for i in parsed_solver_result.tool_calls]
            if any([i is None for i in tool_outputs]):
                content = self.default_reminder_message
                task_state = self._add_eval_message(task_state, solver_result, content=content)
                turn += 1
                continue

            # Add user message containing tool outputs
            task_state = self._add_tool_outputs(task_state, solver_result, tool_outputs)
            turn += 1

        return self._finish_run(task_state, solver_result, None, turn)

    def _get_tool_names_and_descriptions(self, tools: list[Tool]):
        """
        Given sequence of tools, creates a string of each tools name
        and description, each tool's info separated by a newline
        """
        s = ""
        for tool in tools:
            s += f"{tool._name}: {tool._desc}\n"
        return s

    def _parse_solver_result(self, solver_result: SolverResult) -> ParsedSolverResult:
        output = solver_result.output
        tool_calls = self._parse_tool_calls(output)
        final_answer = self._parse_final_answer(output)
        return ParsedSolverResult(tool_calls=tool_calls, final_answer=final_answer)

    def _parse_tool_calls(self, output: str) -> Optional[list[ToolCall]]:
        tool_message_matches = self._find_tool_messages(output)
        if tool_message_matches == []:
            return []

        tool_calls = []
        for tool_name, tool_message in tool_message_matches:
            # Log warning if solver calls a tool that doesn't exist
            try:
                self.name_to_tool[tool_name]
            except KeyError:
                logger.warn(f"Solver tried to call '{tool_name}' tool which doesn't exist!")
                continue

            tool_call = ToolCall(tool_name=tool_name, input=tool_message, output=None)
            tool_calls.append(tool_call)
        return tool_calls

    def _find_tool_messages(self, text: str) -> list[tuple[str, str]]:
        """
        Finds all tool calls, which are formatted [NAME: INPUT],
        where NAME != "Answer" and NAME != "Bugged"
        """
        pattern = r"\(@(?!Answer|Bugged)(\w+): (.+?)\)"
        matches = re.findall(pattern, text, re.DOTALL)
        return matches

    def _parse_final_answer(self, output: str) -> Optional[str]:
        """
        If a final answer exists of form [Answer: OUTPUT], returns the output,
        otherwise returns None
        """
        match = re.search(r"\(@Answer: (.*?)\)", output, re.DOTALL)
        return match.group(1) if match else None

    def _run_tool_call(self, tool_call: ToolCall) -> ToolCall:
        # Prepare task state
        tool_name = tool_call.tool_name
        tool = self.name_to_tool[tool_name]
        tool_input = tool_call.input
        tool_desc = self.name_to_tool[tool_name]._desc

        # Remove quotes if solver wrapped input
        if tool_input.startswith(("'", '"')) and tool_input.endswith(("'", '"')):
            tool_input = tool_input[1:-1]

        task_description = (
            f"Your name is {tool_name}. A description of your purpose is shown below:\n{tool_desc}"
        )
        messages = [Message(role="user", content=tool_input)]
        task_state = ToolTaskState(
            task_description=task_description, messages=messages, current_state=None
        )
        try:
            out = tool(task_state)
        except (TypeError, ValueError, IndexError):
            out = None
        
        if out is None:
            return None

        tool_call.output = out.output
        return tool_call

    def _add_eval_message(
        self,
        task_state: TaskState,
        solver_output: SolverResult,
        content: str,
    ) -> TaskState:
        messages = copy.deepcopy(task_state.messages)
        messages.append(Message(role="assistant", content=solver_output.output))
        # NOTE: we assume that the order of tool_outputs is the same as the order of tool_calls

        messages.append(Message(role="user", content=content))
        new_task_state = TaskState(
            task_description=task_state.task_description,
            messages=messages,
            current_state=None,
        )
        return new_task_state

    def _add_tool_outputs(
        self,
        task_state: TaskState,
        solver_output: SolverResult,
        tool_outputs: list[ToolCall],
    ) -> TaskState:
        content = ""
        for tool_output in tool_outputs:
            name = tool_output.tool_name
            input = tool_output.input
            output = tool_output.output
            content += f"{name} output on input {input}: {output}\n"

        return self._add_eval_message(task_state, solver_output, content)

    def _finish_run(
        self,
        final_task_state: TaskState,
        solver_result: SolverResult,
        final_answer: Optional[str],
        turn: int,
    ) -> RunnerResult:
        expected_answer = self.sample["answer"]
        is_correct = False
        if final_answer is not None:
            final_answer = final_answer.lower().strip()
            # Remove quotes if solver wrapped input
            if final_answer.startswith(("'", '"')) and final_answer.endswith(("'", '"')):
                final_answer = final_answer[1:-1]
            is_correct = final_answer == expected_answer.lower().strip()

        metrics = {
            "is_correct": is_correct,
            "num_turns": turn + 1,  # zero-indexed,
        }

        return RunnerResult(
            final_task_state,
            solver_result,
            metrics,
        )
