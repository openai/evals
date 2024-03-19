import logging
import random
import re
from typing import Any, Sequence, Union

import evals.metrics
from evals.api import CompletionFn
from evals.elsuite.bugged_tools.bugged_tools import ALL_BUGS
from evals.elsuite.bugged_tools.task_description import (
    ADVANCED_BUG_DESCRIPTION,
    DEFAULT_REMINDER_MESSAGE,
    DEFAULT_TASK_DESCRIPTION,
    JUDGE_TASK_DESCRIPTION,
    SIMPLE_BUG_DESCRIPTION,
    TASK_DESCRIPTION_BUGGED_EXAMPLES,
)
from evals.elsuite.bugged_tools.tools import ALL_TOOLS, BuggedTool
from evals.elsuite.bugged_tools.utils import precision_recall_fscore, try_cast_from_str
from evals.elsuite.solver_tools_convo import Runner
from evals.eval import SolverEval
from evals.record import Event
from evals.registry import registry
from evals.solvers.solver import DummySolver, Solver
from evals.task_state import Message

logger = logging.getLogger(__name__)


class BuggedTools(SolverEval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        max_turns: int,
        log_all_metrics: bool,  # Breakdown metrics per tool and per bug
        use_judge: bool,  # Uses judge to identify if solver finds bug
        bug_instructions_type: str,
        *args,
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)

        self.max_turns = max_turns
        self.log_all_metrics = log_all_metrics
        self.use_judge = use_judge

        # Construct the default task description
        task_description_options = {
            "no_warning": DEFAULT_TASK_DESCRIPTION.format(bug_description=""),
            "simple_warning": DEFAULT_TASK_DESCRIPTION.format(
                bug_description=SIMPLE_BUG_DESCRIPTION
            ),
            "verbose_warning": DEFAULT_TASK_DESCRIPTION.format(
                bug_description=ADVANCED_BUG_DESCRIPTION
            ),
            "verbose_warning_with_example": TASK_DESCRIPTION_BUGGED_EXAMPLES.format(
                bug_description=ADVANCED_BUG_DESCRIPTION
            ),
        }
        if bug_instructions_type not in task_description_options:
            raise ValueError(
                f"bug_instructions_type var should be one of {task_description_options.keys()}"
            )
        self.default_task_description = task_description_options[bug_instructions_type]

    def eval_sample(self, solver: Solver, sample: Any, rng: random.Random):
        required_keys = ["task", "answer", "tools", "bugs"]
        assert all([i in sample.keys() for i in required_keys])
        assert isinstance(sample["task"], str)
        assert isinstance(sample["answer"], str)
        assert isinstance(sample["tools"], list)
        assert isinstance(sample["bugs"], dict)

        # Currently this eval assumes one tool
        assert len(sample["tools"]) == 1 and len(sample["bugs"]) <= 1

        # Run eval and record metrics
        name_to_tool = self._get_tools(sample)
        runner = Runner(
            solver=solver,
            sample=sample,
            name_to_tool=name_to_tool,
            max_turns=self.max_turns,
            default_task_description=self.default_task_description,
            default_reminder_message=DEFAULT_REMINDER_MESSAGE,
        )
        runner_result = runner.run()

        final_task_state, final_solver_result, metrics = (
            runner_result.final_task_state,
            runner_result.final_solver_result,
            runner_result.metrics,
        )
        all_messages = final_task_state.messages + [
            Message(role="assistant", content=final_solver_result.output)
        ]

        bugs = [i["bugged_func_name"] for i in sample["bugs"].values()]
        metrics["bugs"] = list(set(bugs))
        metrics["tools"] = sample["tools"]

        # Find if solver predicted bug. Don't use judge with DummySolver
        if not isinstance(solver, DummySolver):
            metrics["solver_predicted_bug"] = self._solver_predicted_bug(sample, all_messages)
        else:
            metrics["solver_predicted_bug"] = False

        # Did solver call tool with the bugged input?
        metrics["solver_used_bugged_input"] = self._solver_used_bugged_input(sample, all_messages)

        evals.record.record_metrics(**metrics)  # type: ignore (evals.record badly hinted)

    def run(self, recorder: evals.record.Recorder) -> dict[str, Union[float, int]]:  # type: ignore (evals.record badly hinted)
        samples = self.get_samples()

        self.eval_all_samples(recorder, samples)
        metrics = recorder.get_metrics()

        solver_didnt_use_bugged_input_rate = len(
            [i for i in metrics if not i["solver_used_bugged_input"]]
        ) / len(metrics)
        task_solved_rate = len([i for i in metrics if i["is_correct"]]) / len(metrics)

        min_num_turns = min([i["num_turns"] for i in metrics])
        max_num_turns = max([i["num_turns"] for i in metrics])
        avg_num_turns = sum([i["num_turns"] for i in metrics]) / len(metrics)

        # Calculate success of solver predicting whether tool was buggy
        tp, fp, tn, fn, accuracy, precision, recall, f1 = precision_recall_fscore(metrics)

        results = {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "task_solved_rate": task_solved_rate,
            "min_num_turns": min_num_turns,
            "max_num_turns": max_num_turns,
            "avg_num_turns": avg_num_turns,
            "solver_didnt_use_bugged_input_rate": solver_didnt_use_bugged_input_rate,
        }

        # Breakdown results per type of tool and bug
        if self.log_all_metrics:
            self._log_additional_metrics(metrics, results)

        return results

    def _log_additional_metrics(self, metrics: Sequence[Event], results: dict):
        """
        Modifies results in-place, breaks results down per tool and per bug
        """
        all_tools = list(set([j for i in metrics for j in i["tools"]]))
        all_bugs = list(set([j for i in metrics for j in i["bugs"]]))

        # Log bug metrics per type of tool
        for tool in all_tools:
            filtered_metrics = [i for i in metrics if i["tools"][0] == tool]
            tp, fp, tn, fn, accuracy, precision, recall, f1 = precision_recall_fscore(
                filtered_metrics
            )

            results[f"tool_{tool}_f1"] = f1
            results[f"tool_{tool}_precision"] = precision
            results[f"tool_{tool}_recall"] = recall
            results[f"tool_{tool}_accuracy"] = accuracy
            results[f"tool_{tool}_tp"] = tp
            results[f"tool_{tool}_fp"] = fp
            results[f"tool_{tool}_tn"] = tn
            results[f"tool_{tool}_fn"] = fn

        # Log bug metrics per type of bug. Only log accuracy since all examples here are positive (bugged)
        for bug in all_bugs:
            filtered_metrics = [i for i in metrics if len(i["bugs"]) > 0]
            filtered_metrics = [i for i in filtered_metrics if i["bugs"][0] == bug]

            tp, fp, tn, fn, accuracy, precision, recall, f1 = precision_recall_fscore(
                filtered_metrics
            )
            results[f"bug_{bug}_accuracy"] = accuracy

    def _get_tools(self, sample: Any) -> dict:
        """
        Creates dict of tools specified in the sample, mapping name of tool to its
        object. Bugs (which are functions) are monkey patched into tools
        """

        # Initialise all tools required
        name_to_tool = {}

        for tool_name in sample["tools"]:
            cls = ALL_TOOLS[tool_name]
            tool = cls()

            # Add bug if specified
            if tool_name in sample["bugs"].keys():
                bugged_input, bugged_func_name = (
                    sample["bugs"][tool._name]["bugged_input"],
                    sample["bugs"][tool._name]["bugged_func_name"],
                )
                bugged_func = ALL_BUGS[bugged_func_name]
                tool = BuggedTool(tool, bugged_input, bugged_func)

            name_to_tool[tool_name] = tool

        return name_to_tool

    def _find_flag_from_assistant(
        self, messages: Sequence[Message], pattern: str
    ) -> list[tuple[str, str]]:
        """
        Finds all flag calls from "assistant" messages with a givern pattern
        """
        all_matches = []
        for m in messages:
            if m.role != "assistant":
                continue

            matches = re.findall(pattern, m.content, re.DOTALL)
            all_matches.extend(matches)
        return all_matches

    def _judge_find_bugs(self, all_messages: Sequence[Message]):
        judge = registry.make_completion_fn("gpt-4-0613", extra_options={"temperature": 0.0})

        # Convert eval system messages to user
        all_messages = [i.to_dict() for i in all_messages if i.role == "assistant"]

        context = (
            [{"role": "system", "content": JUDGE_TASK_DESCRIPTION}]
            + all_messages
            + [{"role": "user", "content": "Has the person identified an error?"}]
        )
        response = judge(context).get_completions()[0]

        if "[YES]" in response:
            return True
        elif "[NO]" in response:
            return False
        else:
            logging.warn(f"Judge didn't output correct flag but instead generated:\n{response}")
            return False

    def _solver_predicted_bug(self, sample: Any, all_messages: Sequence[Message]):
        bugged_pattern = r"\(@Bugged: (.+?)\)"
        predicted_bugged_tools = self._find_flag_from_assistant(all_messages, bugged_pattern)

        if len(predicted_bugged_tools) > 0:
            return True
        elif self.use_judge:
            return self._judge_find_bugs(all_messages)
        else:
            return False

    def _solver_used_bugged_input(self, sample: Any, all_messages: Sequence[Message]):
        """
        Return True if no bug exists or solver called the bugged tool with the bugged input, False otherwise
        """
        if len(sample["bugs"]) == 0:
            return True

        # Assume only one bugged input
        assert len(sample["bugs"]) == 1
        bugged_input = [i["bugged_input"] for i in sample["bugs"].values()][0]

        tool_pattern = r"\(@(?!Answer|Bugged)(\w+): (.+?)\)"
        tool_calls = self._find_flag_from_assistant(all_messages, tool_pattern)

        def strip_and_cast(tool_input, cast_type):
            tool_input = tool_input.strip()
            # Remove quotes if solver wrapped input in "" or ''
            if tool_input.startswith(("'", '"')) and tool_input.endswith(("'", '"')):
                tool_input = tool_input[1:-1]
            return try_cast_from_str(tool_input, cast_type)

        # Get tool inputs and cast to correct type
        tool_inputs_used = [i[1] for i in tool_calls]
        tool_inputs_used = [strip_and_cast(i, type(bugged_input)) for i in tool_inputs_used]
        tool_inputs_used = [i for i in tool_inputs_used if i is not None]

        solver_used_bugged_input = bugged_input in tool_inputs_used
        return solver_used_bugged_input
