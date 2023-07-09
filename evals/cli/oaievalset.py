"""
This file defines the `oaievalset` CLI for running eval sets.
"""
import argparse
import json
import logging
import subprocess
from pathlib import Path
from typing import Optional, cast

from evals.registry import Registry

Task = list[str]
logger = logging.getLogger(__name__)


class Progress:
    def __init__(self, file: str) -> None:
        self.file = Path(file)
        self.completed: list[Task] = []

    def load(self) -> bool:
        if not self.file.exists():
            return False

        with self.file.open() as f:
            for line in f:
                self.completed.append(json.loads(line))
        return len(self.completed) > 0

    def add(self, item: Task) -> None:
        self.completed.append(item)
        self.save()

    def save(self) -> None:
        self.file.parent.mkdir(parents=True, exist_ok=True)
        with self.file.open("w") as f:
            for item in self.completed:
                f.write(json.dumps(item) + "\n")
            print(highlight(f"Saved progress to {self.file}"))


def highlight(str: str) -> str:
    return f"\033[1;32m>>> {str}\033[0m"


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run eval sets through the API")
    parser.add_argument("model", type=str, help="Name of a completion model.")
    parser.add_argument("eval_set", type=str, help="Name of eval set. See registry.")
    parser.add_argument(
        "--registry_path",
        type=str,
        default=None,
        action="append",
        help="Path to the registry",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from last checkpoint.",
    )
    parser.add_argument(
        "--exit-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit if any oaieval command fails.",
    )
    return parser


class OaiEvalSetArguments(argparse.Namespace):
    model: str
    eval_set: str
    registry_path: Optional[str]
    resume: bool
    exit_on_error: bool


def run(
    args: OaiEvalSetArguments,
    unknown_args: list[str],
    registry: Optional[Registry] = None,
    run_command: str = "oaieval",
) -> None:
    registry = registry or Registry()
    if args.registry_path:
        registry.add_registry_paths(args.registry_path)

    commands: list[Task] = []
    eval_set = registry.get_eval_set(args.eval_set) if args.eval_set else None
    if eval_set:
        for index, eval in enumerate(registry.get_evals(eval_set.evals)):
            if not eval or not eval.key:
                logger.debug("The eval #%d in eval_set is not valid", index)

            command = [run_command, args.model, eval.key] + unknown_args
            if args.registry_path:
                command.append("--registry_path")
                command = command + args.registry_path
            if command in commands:
                continue
            commands.append(command)
    else:
        logger.warning("No eval set found for %s", args.eval_set)

    num_evals = len(commands)

    progress = Progress(f"/tmp/oaievalset/{args.model}.{args.eval_set}.progress.txt")
    if args.resume and progress.load():
        print(f"Loaded progress from {progress.file}")
        print(f"{len(progress.completed)}/{len(commands)} evals already completed:")
        for item in progress.completed:
            print("  " + " ".join(item))

    commands = [c for c in commands if c not in progress.completed]
    command_strs = [" ".join(cmd) for cmd in commands]
    print("Going to run the following commands:")
    for command_str in command_strs:
        print("  " + command_str)

    num_already_completed = num_evals - len(commands)
    for idx, command in enumerate(commands):
        real_idx = idx + num_already_completed
        print(highlight("Running command: " + " ".join(command) + f" ({real_idx+1}/{num_evals})"))
        subprocess.run(command, stdout=subprocess.PIPE, check=args.exit_on_error)
        progress.add(command)

    print(highlight("All done!"))


def main() -> None:
    parser = get_parser()
    args, unknown_args = parser.parse_known_args()
    run(cast(OaiEvalSetArguments, args), unknown_args)


if __name__ == "__main__":
    main()
