import logging
import os
import subprocess
from pathlib import Path
from shutil import copyfile
from subprocess import CalledProcessError
from tempfile import TemporaryDirectory
from typing import Callable, Optional

import torch
from openai import OpenAI

from evals.solvers.solver import Solver
from evals.task_state import TaskState

client = OpenAI()
logger = logging.getLogger(__name__)


def complete_text(prompt: str, solver: Solver, **kwargs) -> str:
    """Complete text using the given solver."""

    assert isinstance(solver, Solver)

    prompt = TaskState(task_description=prompt)
    response = solver(prompt, **kwargs)

    return response.output


def get_root_dir() -> Path:
    """Returns the root directory of the repository."""

    return get_parent_dir("evals")


def get_code_dir() -> Path:
    """Returns the `evals/elsuite/hr_ml_agent_bench` directory."""

    return get_root_dir() / "elsuite" / "hr_ml_agent_bench"


def get_data_dir() -> Path:
    """Returns the `evals/registry/data/hr_ml_agent_bench` directory."""

    return get_root_dir() / "registry" / "data" / "hr_ml_agent_bench"


def get_parent_dir(name: str, max_depth: int = 64) -> Path:
    """Returns the parent directory with the given `name`. Only searches up to `max_depth` levels."""

    curdir = Path(__file__).parent

    for _ in range(max_depth):
        if curdir.name == name:
            return curdir

        curdir = curdir.parent

    raise ValueError(f"Couldn't find a parent directory of '{curdir}' named '{name}'!")


def is_gpu_available() -> bool:
    """Returns `True` iff a GPU is available."""

    return torch.cuda.is_available()


def get_gpu_with_most_available_memory() -> Optional[int]:
    """Returns the index of the GPU with the most available memory."""
    try:
        smi_output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total,memory.free",
                "--format=csv,nounits,noheader",
            ],
            encoding="utf-8",
        )
    except (CalledProcessError, FileNotFoundError):
        return None

    max_memory = 0
    gpu_with_max_memory = 0

    for line in smi_output.strip().split("\n"):
        gpu_index, total_memory, free_memory = line.split(", ")
        free_memory = int(free_memory)

        if free_memory > max_memory:
            max_memory = free_memory
            gpu_with_max_memory = gpu_index

    return gpu_with_max_memory


def get_baseline_score(
    baseline_script: Path,
    score_fn: Callable[[Path], float],
    other_files: Optional[list[Path]] = None,
    save_checkpoints: bool = True,
) -> float:
    """
    Executes the `baseline_script` in a temporary directory and returns its score
    using the provided `score_fn`. Optionally, additional files can be provided
    in `other_files` to be copied to the temporary directory. Checkpoints can also
    be saved in the same directory of the `baseline_script` if `save_checkpoints`
    is `True` to avoid re-running computationally expensive baseline scripts.
    """

    assert baseline_script.exists(), f"Expected to find the naive baseline at: {baseline_script}"

    logger.info(f"Executing script: {baseline_script}")

    if other_files is None:
        other_files = []

    for other_file in other_files:
        assert other_file.exists(), f"Expected to find the file at: {other_file}"

    with TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        copyfile(
            src=baseline_script,
            dst=tmp_dir / baseline_script.name,
        )

        for other_file in other_files:
            copyfile(
                src=other_file,
                dst=tmp_dir / other_file.name,
            )

        cmd = ["python", str(baseline_script.name)]
        env = os.environ.copy()
        device = get_gpu_with_most_available_memory()

        if device is not None:
            env["CUDA_VISIBLE_DEVICES"] = device

        with subprocess.Popen(
            args=cmd,
            cwd=tmp_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # combine stderr and stdout
            shell=False,
            text=True,
        ) as process:
            for line in process.stdout:
                logging.info(line.strip())

            # Wait for the process to finish, otherwise the return code
            # may be `None` instead of an integer.
            process.wait()

            assert process.returncode == 0, (
                f"Expected the baseline script {baseline_script} to "
                f"execute successfully, but a return code of: "
                f"{process.returncode}."
            )

        if save_checkpoints:
            for file in tmp_dir.glob("*.checkpoint"):
                dst = baseline_script.parent / file.name

                if dst.exists():
                    continue  # don't overwrite existing files

                logger.info(f"Saving checkpoint for {baseline_script} to {dst}")

                copyfile(
                    src=file,
                    dst=dst,
                )

        score = score_fn(tmp_dir)

    return score
