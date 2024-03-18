"""
This file defines the `Environment` class, which manages the agent's workspace, including files,
datasets, and other resources.

Note: This file is adapted from MLAgentBench with minimal edits made. The original file can be
found at: https://github.com/snap-stanford/MLAgentBench/blob/main/MLAgentBench/environment.py.
"""

import copy
import fnmatch
import json
import os
import shutil
import signal
import time
from logging import getLogger
from multiprocessing import active_children
from pathlib import Path
from traceback import format_exception
from typing import Optional

from dacite import from_dict

from evals.elsuite.ml_agent_bench.high_level_actions import HIGH_LEVEL_ACTIONS
from evals.elsuite.ml_agent_bench.low_level_actions import LOW_LEVEL_ACTIONS
from evals.elsuite.ml_agent_bench.prepare_task import get_research_problem, prepare_task
from evals.elsuite.ml_agent_bench.schema import (
    Action,
    EnhancedJSONEncoder,
    EnvException,
    LLMError,
    Step,
    TooLongPromptError,
    Trace,
)
from evals.solvers.solver import Solver

logger = getLogger(__name__)


class Environment:
    def __init__(
        self,
        log_dir: Path,
        work_dir: Path,
        task: str,
        python_command: str,
        resume: bool,
        resume_step: int,
        device: int,
        max_steps: int,
        max_time: int,
        solver: Solver,
    ):
        self.log_dir = log_dir
        self.work_dir = work_dir
        self.python_command = python_command
        self.resume = resume
        self.resume_step = resume_step
        self.device = device
        self.max_steps = max_steps
        self.max_time = max_time
        self.solver = solver

        self._setup_log_dir()

        self._benchmark_folder_name = task
        self._research_problem = get_research_problem(task)
        self._read_only_files = []
        self._initialize_task_env()  # set up work dir and log dir

        self._action_infos = {t.name: t for t in LOW_LEVEL_ACTIONS + HIGH_LEVEL_ACTIONS}

        self._static_kwargs_for_tools = {
            "device": self.device,
            "python": self.python_command,
            "work_dir": self.work_dir,
            "read_only_files": self.read_only_files,
            "research_problem": self.research_problem,
        }
        self._trace = self._initialize_trace()
        self._start_time = time.time()

    ############################## getters ########################################

    @property
    def research_problem(self):
        return self._research_problem

    @property
    def benchmark_folder_name(self):
        return self._benchmark_folder_name

    @property
    def read_only_files(self):
        return self._read_only_files

    @property
    def action_infos(self):
        return self._action_infos

    @property
    def static_kwargs_for_tools(self):
        return self._static_kwargs_for_tools

    @property
    def trace(self):
        return copy.deepcopy(self._trace)

    @property
    def start_time(self):
        return self._start_time

    ############################## internal functions ########################################

    def _setup_log_dir(self):
        # set up log dir
        if os.path.exists(self.log_dir):
            logger.info(f"log_dir {self.log_dir} already exists")
        else:
            os.makedirs(self.log_dir)

        if os.path.exists(os.path.join(self.log_dir, "tool_logs")):
            logger.info(f"tools_log_dir {os.path.join(self.log_dir, 'tool_logs')} already exists")
        else:
            os.makedirs(os.path.join(self.log_dir, "tool_logs"))

        if os.path.exists(os.path.join(self.log_dir, "traces")):
            logger.info(f"tools_log_dir {os.path.join(self.log_dir, 'traces')} already exists")
        else:
            os.makedirs(os.path.join(self.log_dir, "traces"))

    def _initialize_task_env(self):
        work_dir = self.work_dir

        # remove the workspace folder if it exists
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)

        benchmark_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "benchmarks",
            self.benchmark_folder_name,
        )

        # prepare if there is a prepare.py and it has not been prepared
        prepare_task(benchmark_dir, self.python_command)

        # copy the benchmarks folder to work_dir
        if os.path.exists(os.path.join(benchmark_dir, "env")):
            shutil.copytree(os.path.join(benchmark_dir, "env"), work_dir, symlinks=True)

        # find all read only files
        if os.path.exists(os.path.join(benchmark_dir, "scripts", "read_only_files.txt")):
            ignore_files = (
                open(os.path.join(benchmark_dir, "scripts", "read_only_files.txt"), "r")
                .read()
                .split("\n")
            )
            for path, subdirs, files in os.walk(os.path.join(work_dir)):
                relpath = os.path.relpath(path, work_dir)
                # filter out the files that are read only
                filenames = [os.path.join(relpath, filename) for filename in files]
                for ignore in ignore_files:
                    ignore_filenames = [n for n in filenames if fnmatch.fnmatch(n, ignore)]
                    self.read_only_files.extend(ignore_filenames)

        # init backup folder and remove all content if it exists
        if os.path.exists(os.path.join(work_dir, "backup")):
            shutil.rmtree(os.path.join(work_dir, "backup"))
        os.mkdir(os.path.join(work_dir, "backup"))

        if self.resume:
            shutil.rmtree(work_dir)
            resume_dir = os.path.join(
                self.resume,
                "env_log",
                "traces",
                f"step_{self.resume_step}_files",
            )
            logger.info(f"Restoring workspace ing from {resume_dir}")
            shutil.copytree(resume_dir, work_dir, symlinks=True)
            if not os.path.exists(os.path.join(work_dir, "backup")):
                os.mkdir(os.path.join(work_dir, "backup"))

    def _initialize_trace(self):
        if self.resume:
            logger.info(f"Restoring trace from {self.resume}")
            prev_trace = from_dict(
                data_class=Trace,
                data=json.load(open(os.path.join(self.resume, "env_log", "trace.json"), "r")),
            )
            logger.info(f"Resetting trace to step {self.resume_step}")
            steps = prev_trace.steps[: self.resume_step + 1]
            t = steps[-1].timestamp
            low_level_steps = [s for s in prev_trace.low_level_steps if s.timestamp < t]
            trace = Trace(
                steps=steps,
                low_level_steps=low_level_steps,
                action_infos=self.action_infos,
                task_description=self.research_problem,
            )
        else:
            trace = Trace(
                steps=[],
                low_level_steps=[],
                action_infos=self.action_infos,
                task_description=self.research_problem,
            )
        return trace

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # save error message
        active = active_children()
        logger.info(f"Active Children: {len(active)}")
        # terminate all active children
        for child in active:
            child.terminate()
        # block until all children have closed
        for child in active:
            child.join()
        # report active children
        active = active_children()
        logger.info(f"Active Children: {len(active)}")

        if traceback is not None:
            logger.info("Error message saved in error.txt")
            open(os.path.join(self.log_dir, "error.txt"), "w").write(
                "".join(format_exception(exc_type, exc_value, traceback))
            )
        open(os.path.join(self.log_dir, "overall_time.txt"), "w").write(
            str(time.time() - self.start_time)
        )

    ################################# public functions ########################################

    def is_done(self):
        """Check if the task has reached a final state, either by reaching the maximum steps or time, or because the agent has submitted a final answer."""

        curr_step = len(self.trace.steps)
        # check if any step is final answer
        any_final_answer = any([s.action.name == "Final Answer" for s in self.trace.steps])
        return (
            curr_step >= self.max_steps
            or any_final_answer
            or time.time() - self.start_time > self.max_time
        )

    def execute(self, action: Action, max_seconds_per_step: Optional[int] = None) -> str:
        """Execute an action and return the observation."""

        trace = self._trace

        curr_step = len(trace.steps)
        action_name = action.name
        action_input = action.args

        if action_name == "Final Answer":
            observation = "end"
        elif self.is_done():
            observation = "The environment has shut down because the maximum number of steps or time has been reached. Please submit your final answer."
        elif action_name not in list(self.action_infos.keys()):
            actions = ", ".join(self.action_infos.keys())
            observation = f"Invalid action: {action_name}. Action did not execute. Please use one of the following actions:\n{actions}"
        else:
            # execute the action and get the observation
            log_file = os.path.join(
                os.path.join(self.log_dir, "tool_logs"),
                f"step_{curr_step}_tool_log.log",
            )
            usage = ",\n            ".join(
                [f"{k}: [{v}]" for k, v in self.action_infos[action_name].usage.items()]
            )
            usage = f"""{{
            {usage}
}}"""
            invalid_action_error = f"""No valid action found! Please ensure you're executing a valid action with json inputs. For example, to execute the `List Files` action, you would write:

    Action: List Files
    Action Input: {{
        "dir_path": "."
    }}

Likewise, the input for the action `{action_name}` needs to be valid json with proper entries. Please try again with the correct arguments:

    Action: {action_name}
    Action Input: {usage}"""

            if isinstance(action_input, dict):
                try:
                    if max_seconds_per_step is not None:
                        signal.signal(signal.SIGALRM, _signal_handler)
                        signal.alarm(max_seconds_per_step)

                    observation = self.action_infos[action_name].function(
                        **action_input,
                        log_file=log_file,
                        trace=trace,
                        **self.static_kwargs_for_tools,
                        solver=self.solver,
                    )
                except TooLongPromptError:
                    observation = "EnvError: too long input for the tool"
                except LLMError as e:
                    observation = "LLMError: " + e.message
                except TimeoutError:
                    observation = f"TimeoutError: action execution time exceeded the maximum time limit of {max_seconds_per_step} seconds!"
                except EnvException as e:
                    observation = "EnvError: " + e.message
                except TypeError as e:
                    logger.info(f"Step: {curr_step}")
                    logger.info(e)
                    logger.info(action_input)
                    observation = "EnvError: " + invalid_action_error
                except Exception as e:
                    # should not happen
                    logger.info(f"Step: {curr_step}")
                    logger.info(e)
                    if "Connection aborted." in str(e):
                        raise Exception("Connection aborted for crfm")
                    observation = f"EnvError: Error executing {action_name}."
                finally:
                    if max_seconds_per_step is not None:
                        signal.alarm(0)  # disable the alarm
            else:
                observation = invalid_action_error

        step_time = time.time()

        trace.steps.append(Step(action, observation, step_time))

        self.save(curr_step)

        return observation

    def save(self, curr_step):
        """Save the trace and snapshot of the workspace folder"""
        with open(os.path.join(self.log_dir, "trace.json"), "w") as f:
            json.dump(self.trace, f, indent=4, cls=EnhancedJSONEncoder)

        ##### save a snapshot of the current step
        save_folder = os.path.join(self.log_dir, f"traces/step_{curr_step}_files")
        if os.path.exists(save_folder):
            shutil.rmtree(save_folder)
        os.makedirs(save_folder)

        # save files in the folder that are not read only
        for path, subdirs, files in os.walk(os.path.join(self.work_dir)):
            relpath = os.path.relpath(path, self.work_dir)
            dest = os.path.join(save_folder, relpath)

            for file_name in files:
                file_path = os.path.join(relpath, file_name)
                if file_path not in self.read_only_files:
                    if not os.path.exists(dest):
                        os.makedirs(dest)
                    shutil.copyfile(
                        os.path.join(self.work_dir, file_path),
                        os.path.join(save_folder, file_path),
                    )

    ############## for logging convenience ##############

    def get_task_description(self):
        return self.research_problem, self.benchmark_folder_name

    @property
    def low_level_actions(self):
        return list(filter(lambda x: x.is_primitive, self.action_infos.values()))

    @property
    def high_level_actions(self):
        return list(filter(lambda x: not x.is_primitive, self.action_infos.values()))

    def print_action(self, entries):
        return "".join([k + ": " + v for k, v in entries.items()])


def _signal_handler(signum, frame):
    raise TimeoutError("Time's up! The action exceeded the maximum time limit and terminated early")
