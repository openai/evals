import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from evals.cli import oaievalset


class FakeRegistry:
    def get_eval_set(self, name):
        return SimpleNamespace(evals=["demo.*"])

    def get_evals(self, patterns):
        return [SimpleNamespace(key="demo.dev")]


def make_args(*, exit_on_error: bool):
    return SimpleNamespace(
        model="dummy",
        eval_set="demo-set",
        registry_path=None,
        resume=False,
        exit_on_error=exit_on_error,
    )


def make_progress():
    progress = Mock()
    progress.file = Path("/tmp/test.progress.txt")
    progress.completed = []
    progress.load.return_value = False
    return progress


def test_failed_command_is_not_checkpointed_when_continuing(capsys) -> None:
    progress = make_progress()
    completed_process = subprocess.CompletedProcess(args=[], returncode=1)

    with patch.object(oaievalset, "Progress", return_value=progress), patch.object(
        oaievalset.subprocess, "run", return_value=completed_process
    ) as run_mock:
        oaievalset.run(
            make_args(exit_on_error=False),
            [],
            registry=FakeRegistry(),
            run_command="oaieval",
        )

    run_mock.assert_called_once_with(
        ["oaieval", "dummy", "demo.dev"],
        stdout=subprocess.PIPE,
        check=False,
    )
    progress.add.assert_not_called()
    assert "failed commands were not marked complete" in capsys.readouterr().out


def test_successful_command_is_checkpointed(capsys) -> None:
    progress = make_progress()
    completed_process = subprocess.CompletedProcess(args=[], returncode=0)

    with patch.object(oaievalset, "Progress", return_value=progress), patch.object(
        oaievalset.subprocess, "run", return_value=completed_process
    ):
        oaievalset.run(
            make_args(exit_on_error=False),
            [],
            registry=FakeRegistry(),
            run_command="oaieval",
        )

    progress.add.assert_called_once_with(["oaieval", "dummy", "demo.dev"])
    assert "All done!" in capsys.readouterr().out
