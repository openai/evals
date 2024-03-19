"""Prepare a benchmark folder for a task."""

import json
import os
import subprocess
import sys
from logging import getLogger

from evals.elsuite.hr_ml_agent_bench.utils import get_data_dir

benchmarks_dir = os.path.dirname(os.path.realpath(__file__)) + "/benchmarks"
logger = getLogger(__name__)


def get_research_problem(task: str) -> str:
    """Get the research problem for the given task."""
    
    data_dir = get_data_dir()

    for config in data_dir.glob("**/*.jsonl"):
        with open(config, "r") as f:
            lines = f.readlines()

        for line in lines:
            info = json.loads(line)

            if info["task_name"] != task:
                continue

            assert (
                "research_problem" in info
            ), f"Expected 'research_problem' in {config} for task {task}. Got: {info}."

            return info["research_problem"]

    raise ValueError(f"Task {task} not supported.")


def prepare_task(benchmark_dir, python_command="python"):
    """Run prepare.py in the scripts folder of the benchmark if it exists and has not been run yet."""

    fname_script = os.path.join(benchmark_dir, "scripts", "prepare.py")
    dir_script = os.path.join(benchmark_dir, "scripts", "prepared")

    if not os.path.exists(fname_script):
        return logger.info(f"Not running preparation routine since {fname_script} doesn't exist.")

    if os.path.exists(dir_script):
        return logger.info("prepare.py already prepared")

    logger.info("Running prepare.py...")

    p = subprocess.run(
        args=[python_command, "prepare.py"],
        cwd=os.path.join(benchmark_dir, "scripts"),
    )

    if p.returncode != 0:
        logger.info("prepare.py failed")
        sys.exit(1)

    with open(dir_script, "w") as f:
        f.write("success")

    logger.info("prepare.py finished")
