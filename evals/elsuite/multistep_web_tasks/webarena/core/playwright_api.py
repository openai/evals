"""This file contains classes that represent Playwright objects.
Calling methods on them should call the associated Playwright method
through the Flask app.
"""

import json
import logging
import time
from abc import ABC
from typing import Optional, TypedDict

import docker

from evals.elsuite.multistep_web_tasks.constants import (
    AVAILABLE_SERVICES,
    DEFAULT_PORT_MAPPINGS,
    DOCKER_NAMES,
    FLASK_API_KEY,
    SERVICE_TO_URL,
    URL_TO_SERVICE,
)
from evals.elsuite.multistep_web_tasks.webarena.browser_env.browser_utils import (
    BrowserState,
    BrowserWindowConfig,
)

logger = logging.getLogger(__name__)
# currently commands sometimes fail during navigation
# i want to avoid this
# TODO: find a way to dynamically wait
DEFAULT_WAIT_TIME = 2
N_ALLOWED_RETRIES = 10


class Forwarder(ABC):
    """Class to represent all objects that call
    the Flask Playwright API"""

    def __init__(self, container: docker.models.containers.Container) -> None:  # type: ignore (docker types)
        self.container = container
        self.api_key = FLASK_API_KEY

    def execute_command(self, command: str, n_allowed_attempts: int = 1) -> dict:
        """Execute a command on the Playwright object"""
        data = {"command": command}
        output = self.make_request(
            endpoint="exec_command", data=data, n_allowed_attempts=n_allowed_attempts
        )
        return output

    def make_request(
        self,
        endpoint: str,
        data: Optional[dict] = None,
        n_allowed_attempts: int = 1,
    ) -> dict:
        """Make a request to the Flask app through Docker
        Add some optional retrying for idempotent requests"""
        port = DEFAULT_PORT_MAPPINGS[self.container.name]["internal"]
        url = f"http://localhost:{port}/{endpoint}"

        if data is None:
            data = {}
        data["api-key"] = self.api_key
        json_string = json.dumps(data)
        escaped_json_string = self._escape_quotes_in_json_string(json_string)
        curl_command = (
            f"curl -sS -X POST -H 'Content-Type: application/json' -d '{escaped_json_string}' {url}"
        )
        logger.debug(f"===\n\nCurl command before exec run:\n{curl_command}\n\n===")
        status = None
        n_attempts = 0
        while status != "success" and n_attempts < n_allowed_attempts:
            n_attempts += 1
            raw_output = self.container.exec_run(curl_command)
            if raw_output.exit_code != 0:
                logger.error(f"Request unsuccessful, output is {raw_output}")
                raise ValueError(
                    f"Request unsuccessful, got exec_run exit code {raw_output.exit_code}"
                )
            output = json.loads(raw_output.output)
            if output["status"] != "success":
                logger.error(
                    f"On attempt {n_attempts} of {n_allowed_attempts}, request unsuccessful, output is {output}"
                )
                if n_attempts < n_allowed_attempts:
                    logger.info(f"Retrying request after {DEFAULT_WAIT_TIME} seconds...")
                    time.sleep(DEFAULT_WAIT_TIME)

        return output  # type: ignore (unbound warning but can't be unbound)

    def _double_quotes_to_single_quotes(self, expression: str) -> str:
        """Since we use double quotes *around* the expression in 'evaluate',
        we need to make sure no double quotes appear *inside* the expression"""
        cp = expression
        expression = expression.replace('"', "'")
        if cp != expression:
            logger.warning(f"Changed {cp = } to {expression = }")
        return expression

    def _escape_newlines(self, command: str) -> str:
        """When passing through the API, we need to escape newline chars"""
        return command.replace("\n", "\\n")

    def server_url_to_client_url(self, server_url: str) -> str:
        """Map a server url (i.e. a docker container name (+ port) on the custom bridge network) to
        a client url (e.g. wikipedia.org or reddit.com)
        This is an early version of the function and is liable to break/change"""
        if not server_url.startswith("http://"):
            raise ValueError(f"Expected {server_url = } to start with http://")

        changed_something = False
        for service in AVAILABLE_SERVICES:
            container_name = DOCKER_NAMES[service]["container"]
            container_internal_port = DEFAULT_PORT_MAPPINGS[service]["internal"]
            container_url_no_port = f"http://{container_name}"
            container_url_with_port = f"http://{container_name}:{container_internal_port}"

            # replace with port first, then try without
            if container_url_with_port in server_url:
                if changed_something:
                    logger.warning(f"Changing two things in {server_url}, is this intended?")
                server_url = server_url.replace(container_url_with_port, SERVICE_TO_URL[service])
                changed_something = True
            elif container_url_no_port in server_url:
                if changed_something:
                    logger.warning(f"Changing two things in {server_url}, is this intended?")
                server_url = server_url.replace(container_url_no_port, SERVICE_TO_URL[service])
                changed_something = True

        if not changed_something:
            raise ValueError(f"Could not find a service for server url:\n{server_url}")
        return server_url

    def client_url_to_server_url(self, client_url: str) -> str:
        """Map a a client url (e.g. wikipedia.org or reddit.com) to a server url
        (i.e. a docker container name (+ port) on the custom bridge network)

        This is an early version of the function and is liable to break/change"""
        if not client_url.startswith("http://"):
            raise ValueError(f"Expected {client_url = } to start with http://")

        changed_something = False
        for url in URL_TO_SERVICE:
            service = URL_TO_SERVICE[url]
            if url in client_url:
                if changed_something:
                    logger.warning(f"Changing two things in {client_url}, is this intended?")
                container_name = DOCKER_NAMES[service]["container"]
                container_internal_port = DEFAULT_PORT_MAPPINGS[service]["internal"]
                container_url_with_port = f"http://{container_name}:{container_internal_port}"
                client_url = client_url.replace(url, container_url_with_port)
                changed_something = True
        if not changed_something:
            raise ValueError(f"Could not find a service for client url:\n{client_url}")
        return client_url

    def _process_response(self, output: dict) -> Optional[dict]:
        if output["status"] != "success":
            raise ValueError(f"Request unsuccessful, got output {output}")
        self.server_url = output["url"]
        return output.get("content", None)

    def _escape_quotes_in_json_string(self, json_string: str) -> str:
        return json_string.replace("'", "'\\''")


class PageForwarder(Forwarder):
    """Class to represent a Playwright Page object"""

    def __init__(
        self,
        container: docker.models.containers.Container,  # type: ignore
        viewport_size: "ViewportSize",
    ) -> None:
        super().__init__(container)
        self.server_url = ""
        self.viewport_size = viewport_size
        self.client = ClientForwarder(self)
        self.mouse = MouseForwarder(self)
        self.keyboard = KeyboardForwarder(self)

    @property
    def url(self) -> str:
        if self.server_url == "":
            return ""
        else:
            return self.server_url_to_client_url(self.server_url)

    def setup(self) -> None:
        """Not sure if this should go in PageForwarder or some BrowserForwarder
        class or what, but it's here for now"""
        # call the setup endpoint and let the flask app set itself up
        out = self.make_request(endpoint="setup")
        if out["status"] != "success":
            raise ValueError(f"setup failed with output {out}")

    def shutdown(self) -> None:
        """Not sure if this should go in PageForwarder or some BrowserForwarder
        class or what, but it's here for now"""
        # call the shutdown endpoint and let the flask app handle shuttind down
        out = self.make_request(endpoint="shutdown")
        if out["status"] != "success":
            raise ValueError(f"shutdown failed with output {out}")

    def content(self) -> str:
        """Get the html content of the page"""
        out = self.execute_command("page.content()", n_allowed_attempts=N_ALLOWED_RETRIES)
        rv = self._process_response(out)
        assert isinstance(rv, str)
        return rv

    def goto(self, url: str) -> None:
        """NOTE: we handle conversion of urls from client to server here
        (and in the other methods that take urls) rather than in execute_command,
        since we don't know which parts of a command are urls"""
        logger.info(f"===\n{self.url = } before goto\n===")
        logger.info(f"===\n\nGoing to client url {url}\n\n---")
        try:
            url = self.client_url_to_server_url(url)
        # if the url is invalid, don't go anywhere
        except ValueError:
            logger.error(
                f"Could not convert {url = } to server url (is it part of the open internet?)"
            )
            return
        logger.info(f"---\n\nGoing to server url {url}\n\n===")
        out = self.execute_command(command=f"page.goto('{url}')")
        self._process_response(out)
        logger.info(f"===\n{self.url = } after goto\n===")

    def title(self) -> str:
        out = self.execute_command("page.title()", n_allowed_attempts=N_ALLOWED_RETRIES)
        rv = self._process_response(out)
        assert isinstance(rv, str)
        return rv

    def evaluate(self, expression: str) -> str:
        modified_expression = self._double_quotes_to_single_quotes(expression)
        out = self.execute_command(f"""page.evaluate("{modified_expression}")""")
        rv = self._process_response(out)
        assert isinstance(rv, str)
        return rv

    def go_back(self) -> None:
        out = self.execute_command("page.go_back()")
        self._process_response(out)

    def go_forward(self) -> None:
        out = self.execute_command("page.go_forward()")
        self._process_response(out)

    def fetch_domtree(self) -> dict:
        assert self.client is not None
        tree = self.client.send(
            "DOMSnapshot.captureSnapshot",
            {
                "computedStyles": [],
                "includeDOMRects": True,
                "includePaintOrder": True,
            },
        )

        # calibrate the bounds, in some cases, the bounds are scaled somehow
        bounds = tree["documents"][0]["layout"]["bounds"]
        b = bounds[0]
        n = b[2] / self.viewport_size["width"]
        bounds = [[x / n for x in bound] for bound in bounds]
        tree["documents"][0]["layout"]["bounds"] = bounds
        # add union bound placeholder
        tree["documents"][0]["layout"]["unionBounds"] = [None for _ in bounds]

        return tree

    def fetch_browser_window_config(self) -> BrowserWindowConfig:
        # extract browser info
        desired_properties = [
            "page.evaluate('window.pageYOffset')",
            "page.evaluate('window.pageXOffset')",
            "page.evaluate('window.screen.width')",
            "page.evaluate('window.screen.height')",
            "page.evaluate('window.devicePixelRatio')",
        ]

        output = self.make_request("exec_commands", {"commands": desired_properties})

        retrieved_properties = output["content"]
        assert retrieved_properties is not None

        win_width = retrieved_properties["page.evaluate('window.screen.width')"]
        win_height = retrieved_properties["page.evaluate('window.screen.height')"]
        x_offset = retrieved_properties["page.evaluate('window.pageXOffset')"]
        y_offset = retrieved_properties["page.evaluate('window.pageYOffset')"]
        browser_config: BrowserWindowConfig = {
            "win_upper_bound": x_offset,
            "win_left_bound": y_offset,
            "win_width": win_width,
            "win_height": win_height,
            "win_right_bound": x_offset + win_width,
            "win_lower_bound": y_offset + win_height,
            "device_pixel_ratio": retrieved_properties["page.evaluate('window.devicePixelRatio')"],
        }
        assert (
            browser_config["device_pixel_ratio"] == 1.0
        ), f"device_pixel_ratio is {browser_config['device_pixel_ratio']}, should be 1.0"

        # casting to BrowserWindowConfig TypedDict
        return browser_config

    def fetch_browser_info(self) -> BrowserState:
        tree = self.fetch_domtree()
        config: BrowserWindowConfig = self.fetch_browser_window_config()
        return BrowserState({"DOMTree": tree, "config": config})

    def wait_for_load_state(self, state: str, timeout: int = 500) -> None:
        tic = time.perf_counter()
        out = self.execute_command(f"page.wait_for_load_state(state='{state}', timeout={timeout})")
        self._process_response(out)
        toc = time.perf_counter()
        logger.info(f"wait_for_load_state for '{state}' took {toc - tic:0.4f} seconds")
        logger.info(f"\n====\n\noutput from wait_for_load_state:\n{out}\n\n====\n")

    def wait_for_event(self, event: str, timeout: int = 500) -> None:
        tic = time.perf_counter()
        out = self.execute_command(f"page.wait_for_event(event='{event}', timeout={timeout})")
        self._process_response(out)
        toc = time.perf_counter()
        logger.info(f"wait_for_event for '{event}' took {toc - tic:0.4f} seconds")


class ClientForwarder(Forwarder):
    """Class to represent a Playwright CDPSession object"""

    def __init__(self, page: PageForwarder) -> None:
        super().__init__(page.container)
        self.page = page

    def send(self, method: str, params: dict) -> dict:
        """Send a command to the CDPSession"""
        out = self.execute_command(f"client.send(method='{method}', params={params})")
        rv = self._process_response(out)
        assert isinstance(rv, dict)
        return rv


class MouseForwarder(Forwarder):
    def __init__(self, page: PageForwarder) -> None:
        super().__init__(page.container)
        self.page = page

    def click(self, x: float, y: float) -> None:
        out = self.execute_command(f"page.mouse.click({x}, {y})")
        self._process_response(out)

    def move(self, x: float, y: float) -> None:
        out = self.execute_command(f"page.mouse.move({x}, {y})")
        self._process_response(out)


class KeyboardForwarder(Forwarder):
    def __init__(self, page: PageForwarder) -> None:
        super().__init__(page.container)
        self.page = page

    def type(self, text: str) -> None:
        escaped_text = self._escape_newlines(text)
        modified_text = self._double_quotes_to_single_quotes(escaped_text)
        out = self.execute_command(f"""page.keyboard.type("{modified_text}")""")
        self._process_response(out)

    def press(self, key: str) -> None:
        out = self.execute_command(f"page.keyboard.press('{key}')")
        self._process_response(out)


class ViewportSize(TypedDict):
    width: int
    height: int
