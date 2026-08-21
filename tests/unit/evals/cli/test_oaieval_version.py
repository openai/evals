import os

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test")

from evals.cli import oaieval


def test_version_flag_prints_package_version(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(oaieval, "_package_version", lambda: "9.8.7")

    with pytest.raises(SystemExit) as exc_info:
        oaieval.get_parser().parse_args(["--version"])

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == "9.8.7"


def test_package_version_falls_back_when_distribution_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_distribution(name: str) -> str:
        raise oaieval.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(oaieval.importlib.metadata, "version", missing_distribution)

    assert oaieval._package_version() == "unknown"
