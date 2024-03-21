import base64
import io
import json
import re
from pathlib import Path
from typing import Any, Optional

from beartype import beartype
from PIL import Image

from evals.elsuite.multistep_web_tasks.webarena.browser_env import (
    BrowserAction,
    BrowserActionTypes,
    ObservationMetadata,
    action2str,
)
from evals.elsuite.multistep_web_tasks.webarena.browser_env.browser_utils import BrowserEnvOutput

HTML_TEMPLATE = """
<!DOCTYPE html>
<head>
    <style>
        pre {{
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
    </style>
</head>
<html>
    <body>
     {body}
    </body>
</html>
"""


@beartype
def get_render_action(
    action: BrowserAction,
    observation_metadata: dict[str, ObservationMetadata],
    action_set_tag: str,
) -> str:
    """Parse the predicted actions for rendering purpose. More comprehensive information"""
    if action_set_tag == "id_accessibility_tree":
        text_meta_data = observation_metadata["text"]
        if action.data["element_id"] in text_meta_data["obs_nodes_info"]:
            node_content = text_meta_data["obs_nodes_info"][action.data["element_id"]]["text"]
        else:
            node_content = "No match found"

        action_str = f"<div class='raw_parsed_prediction' style='background-color:grey'><pre>{action.data['raw_prediction']}</pre></div>"
        action_str += f"<div class='action_object' style='background-color:grey'><pre>{repr(action)}</pre></div>"
        action_str += f"<div class='parsed_action' style='background-color:yellow'><pre>{action2str(action, action_set_tag, node_content)}</pre></div>"

    elif action_set_tag == "playwright":
        action_str = action.data["pw_code"]
    else:
        raise ValueError(f"Unknown action type {action.data['action_type']}")
    return action_str


@beartype
def get_action_description(
    action: BrowserAction,
    observation_metadata: dict[str, ObservationMetadata],
    action_set_tag: str,
    prompt_constructor: Optional[Any],
) -> str:
    """Generate the text version of the predicted actions to store in action history for prompt use.
    May contain hint information to recover from the failures"""

    if action_set_tag == "id_accessibility_tree":
        text_meta_data = observation_metadata["text"]
        if action.data["action_type"] in [
            BrowserActionTypes.CLICK,
            BrowserActionTypes.HOVER,
            BrowserActionTypes.TYPE,
        ]:
            action_name = str(action.data["action_type"]).split(".")[1].lower()
            if action.data["element_id"] in text_meta_data["obs_nodes_info"]:
                node_content = text_meta_data["obs_nodes_info"][action.data["element_id"]]["text"]
                node_content = " ".join(node_content.split()[1:])
                action_str = action2str(action, action_set_tag, node_content)
            else:
                action_str = f"Attempt to perfom \"{action_name}\" on element \"[{action.data['element_id']}]\" but no matching element found. Please check the observation more carefully."
        else:
            if (
                action.data["action_type"] == BrowserActionTypes.NONE
                and prompt_constructor is not None
            ):
                action_splitter = prompt_constructor.agent_config.action_splitter
                action_str = f'The previous prediction you issued was "{action.data["raw_prediction"]}". However, the format was incorrect. Ensure that the action is wrapped inside a pair of {action_splitter} and enclose arguments within [] as follows: {action_splitter}action [arg] ...{action_splitter}.'
            else:
                action_str = action2str(action, action_set_tag, "")

    elif action_set_tag == "playwright":
        action_str = action.data["pw_code"]

    else:
        raise ValueError(f"Unknown action type {action.data['action_type']}")

    return action_str


class RenderHelper(object):
    """Helper class to render text and image observations and meta data in the trajectory"""

    def __init__(self, config_file: str, result_dir: str, action_set_tag: str) -> None:
        with open(config_file, "r") as f:
            _config = json.load(f)
            _config_str = ""
            for k, v in _config.items():
                _config_str += f"{k}: {v}\n"
            _config_str = f"<pre>{_config_str}</pre>\n"
            task_id = _config["task_id"]

        self.action_set_tag = action_set_tag

        self.render_file = open(Path(result_dir) / f"render_{task_id}.html", "a+")
        self.render_file.truncate(0)
        # write init template
        self.render_file.write(HTML_TEMPLATE.format(body=f"{_config_str}"))
        self.render_file.read()
        self.render_file.flush()

    def render(
        self,
        action: BrowserAction,
        env_output: BrowserEnvOutput,
        meta_data: dict[str, Any],
        render_screenshot: bool = False,
    ) -> None:
        """Render the trajectory"""
        # text observation
        observation = env_output.observation
        text_obs = observation.acctree
        info = env_output.info
        new_content = "<h2>New Page</h2>\n"
        new_content += f"<h3 class='url'><a href={info.page.url}>URL: {info.page.url}</a></h3>\n"
        new_content += f"<div class='state_obv'><pre>{text_obs}</pre><div>\n"

        if render_screenshot:
            # image observation
            img_obs = observation.image
            image = Image.fromarray(img_obs)
            byte_io = io.BytesIO()
            image.save(byte_io, format="PNG")
            byte_io.seek(0)
            image_bytes = base64.b64encode(byte_io.read())
            image_str = image_bytes.decode("utf-8")
            new_content += (
                f"<img src='data:image/png;base64,{image_str}' style='width:50vw; height:auto;'/>\n"
            )

        # meta data
        new_content += f"<div class='prev_action' style='background-color:pink'>{meta_data['action_history'][-1]}</div>\n"

        # action
        action_str = get_render_action(
            action,
            info.observation_metadata,
            action_set_tag=self.action_set_tag,
        )
        # with yellow background
        action_str = f"<div class='predict_action'>{action_str}</div>"
        new_content += f"{action_str}\n"

        # add new content
        self.render_file.seek(0)
        html = self.render_file.read()
        html_body = re.findall(r"<body>(.*?)</body>", html, re.DOTALL)[0]
        html_body += new_content

        html = HTML_TEMPLATE.format(body=html_body)
        self.render_file.seek(0)
        self.render_file.truncate()
        self.render_file.write(html)
        self.render_file.flush()

    def close(self) -> None:
        self.render_file.close()
