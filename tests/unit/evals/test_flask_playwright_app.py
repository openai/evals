import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "evals"
    / "elsuite"
    / "multistep_web_tasks"
    / "docker"
    / "flask-playwright"
    / "app.py"
)
SPEC = importlib.util.spec_from_file_location("flask_playwright_app", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
flask_playwright_app = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(flask_playwright_app)


def test_execute_command_redacts_exception_details_and_escapes_log(caplog) -> None:
    command = "1 / 0\nforged-log-entry"

    with flask_playwright_app.app.app_context(), caplog.at_level(
        logging.INFO, logger=flask_playwright_app.logger.name
    ):
        with pytest.raises(ValueError) as exc_info:
            flask_playwright_app._execute_command({"command": command})

    assert exc_info.value.args[0] == "Error executing command"
    assert exc_info.value.args[1].get_json() == {
        "status": "error",
        "message": "error executing command",
    }
    command_log = next(
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("Error executing command:")
    )
    assert "\\nforged-log-entry" in command_log
    assert "\nforged-log-entry" not in command_log


def test_setup_does_not_return_exception_text(monkeypatch) -> None:
    def fail_setup():
        raise RuntimeError("sensitive setup detail")

    monkeypatch.setattr(flask_playwright_app, "sync_playwright", fail_setup)
    monkeypatch.setattr(flask_playwright_app, "playwright", None)
    monkeypatch.setattr(flask_playwright_app, "browser", None)
    monkeypatch.setattr(flask_playwright_app, "page", None)
    monkeypatch.setattr(flask_playwright_app, "client", None)

    response = flask_playwright_app.app.test_client().post(
        "/setup",
        json={"api-key": flask_playwright_app.FLASK_API_KEY},
    )

    payload = response.get_json()
    assert payload == {
        "status": "error",
        "message": "failed to start session (already started?)",
    }
    assert "sensitive setup detail" not in response.get_data(as_text=True)


def test_exec_command_does_not_return_serialization_exception(monkeypatch) -> None:
    monkeypatch.setattr(
        flask_playwright_app,
        "page",
        SimpleNamespace(url="https://example.com"),
    )
    monkeypatch.setattr(flask_playwright_app, "_execute_command", lambda _json: object())

    response = flask_playwright_app.app.test_client().post(
        "/exec_command",
        json={
            "api-key": flask_playwright_app.FLASK_API_KEY,
            "command": "return-object",
        },
    )

    assert response.get_json() == {
        "status": "success",
        "message": "could not return results of executed command",
        "content": None,
        "url": "https://example.com",
    }
