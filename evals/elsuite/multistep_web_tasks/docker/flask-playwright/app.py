"""Simple flask app for running Playwright commands inside a docker container.
Known issues:
- Using 'exec' is not that secure, but since only our application can call this API,
it should be fine (plus the model can execute arbitrary code in this network anyway)
- The request handling is pretty messy currently, and I check the request for None a lot
    - I'm sure there's a cleaner way to structure the app
- Playwright (as I'm using it) is not thread-safe, so I'm running single-threaded
"""
import logging

from flask import Flask, jsonify, request
from playwright.sync_api import ViewportSize, sync_playwright

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP_PORT = 8507
app = Flask(__name__)
playwright = None
browser = None
page = None
client = None

# NOTE: this is just to prevent the model from calling this API
# from inside the docker network (since it won't know the key).
# We can't import this from constants.py because once dockerized it won't have access
FLASK_API_KEY = "key-FLASKPLAYWRIGHTKEY"

# TODO: pass this instead of hardcoding it
VIEWPORT_SIZE = ViewportSize({"width": 1280, "height": 720})


@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "success", "message": "flask-playwright"})


@app.route("/setup", methods=["POST"])
def setup():
    api_key_present = ensure_api_key(request)
    if not api_key_present:
        return jsonify({"status": "error", "message": "no/bad api key"})

    global playwright, browser, page, client
    try:
        assert playwright is None, "playwright should be None"
        assert browser is None, "browser should be None"
        assert page is None, "page should be None"
        assert client is None, "client should be None"
        context_manager = sync_playwright()
        playwright = context_manager.__enter__()
        browser = playwright.chromium.launch(headless=True)
        browser_context = browser.new_context(
            viewport=VIEWPORT_SIZE,
            storage_state=None,  # TODO: pass this if needed (how to handle auth?)
            device_scale_factor=1,
        )
        page = browser_context.new_page()
        client = page.context.new_cdp_session(page)  # talk to chrome devtools
        client.send("Accessibility.enable")  # to get AccessibilityTrees
    except Exception as e:
        logger.error("Failed to start session", exc_info=True)
        return jsonify(
            {"status": "error", "message": "failed to start session (already started?)"}
        )
    return jsonify({"status": "success", "message": "session started"})


@app.route("/shutdown", methods=["POST"])
def shutdown():
    """Shut everything down and clear variables, so this container can be reused"""
    global playwright, browser, page, client
    if browser is None or playwright is None:
        return jsonify({"status": "error", "message": "no session started"})
    try:
        browser.close()
        playwright.stop()
        playwright = None
        browser = None
        page = None
        client = None
    except Exception:
        return jsonify({"status": "error", "message": "failed to end session (already ended?)"})
    return jsonify({"status": "success", "message": "session ended"})


@app.route("/exec_command", methods=["POST"])
def exec_command():
    api_key_present = ensure_api_key(request)
    if not api_key_present:
        return jsonify({"status": "error", "message": "no api key"})

    if request.json is None:
        return jsonify({"status": "error", "message": "no json data"})

    command = request.json.get("command", None)
    if command is None:
        return jsonify({"status": "error", "message": "no command"})

    global page
    if page is None:
        return jsonify({"status": "error", "message": "no session started"})

    try:
        result = _execute_command(request.json)
    except ValueError as e:
        assert len(e.args) == 2, "ValueError should have a message and a return object"
        logger.error(e.args[0])
        return e.args[1]
    try:
        response = jsonify(
            {
                "status": "success",
                "message": f"executed command {request.json['command']}",
                "content": result,
                "url": page.url,
            }
        )
    except TypeError as e:
        logger.error(f"Error returning results for command: {request.json['command']}: {e}")
        response = jsonify(
            {
                "status": "success",
                "message": f"An internal error has occurred while returning the results",
                "content": str(e),
                "url": page.url,
            }
        )

    return response


@app.route("/exec_commands", methods=["POST"])
def exec_commands():
    api_key_present = ensure_api_key(request)
    if not api_key_present:
        return jsonify({"status": "error", "message": "no api key"})

    if request.json is None:
        return jsonify({"status": "error", "message": "no json data"})

    commands = request.json.get("commands", None)
    if commands is None:
        return jsonify({"status": "error", "message": "no commands"})

    global page
    if page is None:
        return jsonify({"status": "error", "message": "no session started"})

    try:
        results = _execute_commands(request.json)
    except ValueError as e:
        assert len(e.args) == 2, "ValueError should have a message and a return object"
        logger.error(e.args[0])
        return e.args[1]
    try:
        response = jsonify(
            {
                "status": "success",
                "message": f"executed commands {request.json['commands']}",
                "content": results,
                "url": page.url,
            }
        )
    except TypeError as e:
        logger.error(f"Error returning results of executed commands: {e}")
        response = jsonify(
            {
                "status": "success",
                "message": f"An internal error has occurred while returning the results",
                "content": str(e),
                "url": page.url,
            }
        )
    return response


def _execute_command(json_data: dict):
    # NOTE: This is definitely Not Safe, but the only thing that should be able to call this
    # is my own code
    global playwright, browser, page, client
    command = json_data.get("command", None)
    if command is None:
        raise ValueError("No command", jsonify({"status": "error", "message": "no command"}))

    try:
        result = eval(command)
        return result
    except Exception as e:
        logger.info(f"Error executing command: {command}")
        logger.error(e)
        raise ValueError(
            f"Error executing command {command}",
            jsonify({"status": "error", "message": f"An internal error has occurred while executing the command"}),
        )


def _execute_commands(json_data: dict):
    results = {}
    for command in json_data["commands"]:
        try:
            results[command] = _execute_command({"command": command})
        except ValueError as e:
            # maybe we want to handle this in a more fancy way later
            raise e
    return results


def ensure_api_key(request):
    # NOTE: this is just to prevent the model from calling this API
    if request.json is None:
        return False

    if request.json.get("api-key", None) != FLASK_API_KEY:
        return False

    return True


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=APP_PORT, threaded=False)
