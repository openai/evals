"""
Browser Env action space.
Inspited by Farama-Foundation/miniwob-plusplus
"""

import ast
import logging
import random
import re
import string
from dataclasses import dataclass
from enum import IntEnum
from itertools import chain
from typing import Any, Optional, TypedDict, Union, cast

import numpy as np
import numpy.typing as npt
from beartype import beartype
from beartype.door import is_bearable
from gymnasium import spaces
from playwright._impl._api_structures import ViewportSize
from playwright.async_api import Locator as ALocator
from playwright.sync_api import Locator

from evals.elsuite.multistep_web_tasks.webarena.bash_env.actions import (
    BashAction,
    BashCommandAction,
    BashStopAction,
)
from evals.elsuite.multistep_web_tasks.webarena.browser_env.constants import (
    ASCII_CHARSET,
    FREQ_UNICODE_CHARSET,
    MAX_ANSWER_LENGTH,
    MAX_ELEMENT_ID,
    MAX_ELEMENT_INDEX_IN_VIEWPORT,
    MAX_PAGE_NUMBER,
    MAX_VANILLA_STR_LENGTH,
    PLAYWRIGHT_ACTIONS,
    PLAYWRIGHT_LOCATORS,
    ROLES,
    SPECIAL_KEY_MAPPINGS,
    SPECIAL_KEYS,
    SPECIAL_LOCATORS,
    TEXT_MAX_LENGTH,
    TYPING_MAX_LENGTH,
    URL_MAX_LENGTH,
    RolesType,
)
from evals.elsuite.multistep_web_tasks.webarena.browser_env.processors import ObservationProcessor
from evals.elsuite.multistep_web_tasks.webarena.core.env import Action, ParsingErrorAction
from evals.elsuite.multistep_web_tasks.webarena.core.playwright_api import PageForwarder

logger = logging.getLogger(__name__)


class ParsedPlaywrightCode(TypedDict):
    function_name: str
    arguments: list[str]
    keywords: dict[str, Any]


@beartype
def is_in_viewport(element: Locator, viewport: ViewportSize, threshold: float = 0.3) -> bool:
    """Given a playwright locator, check if it is in the viewport"""
    box = element.bounding_box()
    assert box is not None
    boxx0 = box["x"]
    boxx1 = box["x"] + box["width"]
    boxy0 = box["y"]
    boxy1 = box["y"] + box["height"]
    viewportx0, viewporty0 = 0, 0
    viewportx1, viewporty1 = viewport["width"], viewport["height"]
    inter = max(0, min(boxx1, viewportx1) - max(boxx0, viewportx0)) * max(
        0, min(boxy1, viewporty1) - max(boxy0, viewporty0)
    )
    ratio = inter / (box["width"] * box["height"])
    return ratio > threshold


@beartype
async def async_is_in_viewport(
    element: ALocator, viewport: ViewportSize, threshold: float = 0.3
) -> bool:
    box = await element.bounding_box()
    assert box is not None
    boxx0 = box["x"]
    boxx1 = box["x"] + box["width"]
    boxy0 = box["y"]
    boxy1 = box["y"] + box["height"]
    viewportx0, viewporty0 = 0, 0
    viewportx1, viewporty1 = viewport["width"], viewport["height"]
    inter = max(0, min(boxx1, viewportx1) - max(boxx0, viewportx0)) * max(
        0, min(boxy1, viewporty1) - max(boxy0, viewporty0)
    )
    ratio = inter / (box["width"] * box["height"])
    return ratio > threshold


class BrowserActionDict(TypedDict):
    action_type: int
    coords: npt.NDArray[np.float32]
    element_role: int
    element_name: str
    text: list[int]
    page_number: int
    url: str
    nth: int
    element_id: str
    direction: str
    key_comb: str
    pw_code: str
    answer: str
    raw_prediction: str  # raw prediction from the model


@dataclass
class BrowserAction(Action):
    data: BrowserActionDict


@beartype
def action2str(
    action: Union[BrowserAction, None], action_set_tag: str, semantic_element: str = ""
) -> str:
    """Return the string representation of an action

    sementic_element: the semantic information of the element
    such as a line in an accessibility tree
    """
    # if the action is None, then just return "None"
    if action is None:
        return "None"
    if action_set_tag == "id_accessibility_tree":
        element_id = action.data["element_id"]

        # this used to be a match statement, changed for 3.9 compatibility
        action_type = action.data["action_type"]
        if action_type == BrowserActionTypes.CLICK:
            # [ID=X] xxxxx
            action_str = f"click [{element_id}] where [{element_id}] is {semantic_element}"

        elif action_type == BrowserActionTypes.TYPE:
            text = "".join([_id2key[i] for i in action.data["text"]])
            action_str = f"type [{element_id}] [{text}] where [{element_id}] is {semantic_element}"

        elif action_type == BrowserActionTypes.HOVER:
            action_str = f"hover [{element_id}] where [{element_id}] is {semantic_element}"

        elif action_type == BrowserActionTypes.SCROLL:
            action_str = f"scroll [{action.data['direction']}]"

        elif action_type == BrowserActionTypes.KEY_PRESS:
            action_str = f"press [{action.data['key_comb']}]"

        elif action_type == BrowserActionTypes.GOTO_URL:
            action_str = f"goto [{action.data['url']}]"

        elif action_type == BrowserActionTypes.NEW_TAB:
            action_str = "new_tab"

        elif action_type == BrowserActionTypes.PAGE_CLOSE:
            action_str = "close_tab"

        elif action_type == BrowserActionTypes.GO_BACK:
            action_str = "go_back"

        elif action_type == BrowserActionTypes.GO_FORWARD:
            action_str = "go_forward"

        elif action_type == BrowserActionTypes.PAGE_FOCUS:
            action_str = f"page_focus [{action.data['page_number']}]"

        elif action_type == BrowserActionTypes.STOP:
            action_str = f"stop [{action.data['answer']}]"

        elif action_type == BrowserActionTypes.NONE:
            action_str = "none"

        else:
            raise ValueError(f"Unknown action type {action.data['action_type']}")

    else:
        raise NotImplementedError(f"Unknown action set tag {action_set_tag}")

    return action_str


def action2create_function(action: BrowserAction) -> str:
    # this used to be a match statement, changed for 3.9 compatibility
    action_type = action.data["action_type"]
    if action_type == BrowserActionTypes.NONE:
        return "create_none_action()"

    # mouse wheel and keyboard action
    elif action_type == BrowserActionTypes.SCROLL:
        direction = "up" if "up" in action.data["direction"] else "down"
        return f"create_scroll_action({repr(direction)})"
    elif action_type == BrowserActionTypes.KEY_PRESS:
        return f"create_key_press_action({repr(action.data['key_comb'])})"
    # inter-page actions
    elif action_type == BrowserActionTypes.PAGE_FOCUS:
        return f"create_page_focus_action({action.data['page_number']})"
    elif action_type == BrowserActionTypes.NEW_TAB:
        return "create_new_tab_action()"
    elif action_type == BrowserActionTypes.GO_BACK:
        return "create_go_back_action()"
    elif action_type == BrowserActionTypes.GO_FORWARD:
        return "create_go_forward_action()"
    elif action_type == BrowserActionTypes.GOTO_URL:
        return f"create_goto_url_action({repr(action.data['url'])})"
    elif action_type == BrowserActionTypes.PAGE_CLOSE:
        return "create_page_close_action()"

    # low-level keyboard and mouse actions
    elif action_type == BrowserActionTypes.MOUSE_CLICK:
        return f"create_mouse_click_action({action.data['coords'][0]}, {action.data['coords'][1]})"
    elif action_type == BrowserActionTypes.MOUSE_HOVER:
        return f"create_mouse_hover_action({action.data['coords'][0]}, {action.data['coords'][1]})"
    elif action_type == BrowserActionTypes.KEYBOARD_TYPE:
        return (
            f"create_keyboard_type_action({list(map(lambda x: _id2key[x], action.data['text']))})"
        )

    # mid-level keyboard and mouse actions
    elif action_type == BrowserActionTypes.CLICK:
        args = []
        args.append(f"element_id={repr(action.data['element_id'])}")
        args.append(f"element_role={repr(_id2role[action.data['element_role']])}")
        args.append(f"element_name={repr(action.data['element_name'])}")
        args.append(f"pw_code={repr(action.data['pw_code'])}")
        args_str = ", ".join(args)
        return f"create_click_action({args_str})"
    elif action_type == BrowserActionTypes.HOVER:
        args = []
        args.append(f"element_id={repr(action.data['element_id'])}")
        args.append(f"element_role={repr(_id2role[action.data['element_role']])}")
        args.append(f"element_name={repr(action.data['element_name'])}")
        args.append(f"pw_code={repr(action.data['pw_code'])}")
        args_str = ", ".join(args)
        return f"create_hover_action({args_str})"
    elif action_type == BrowserActionTypes.TYPE:
        args = []
        text = "".join(map(lambda x: _id2key[x], action.data["text"]))
        args.append(f"text={repr(text)}")
        args.append(f"element_id={repr(action.data['element_id'])}")
        args.append(f"element_role={repr(_id2role[action.data['element_role']])}")
        args.append(f"element_name={repr(action.data['element_name'])}")
        args.append(f"pw_code={repr(action.data['pw_code'])}")
        args_str = ", ".join(args)
        return f"create_type_action({args_str})"

    # high-level actions, only support locators from playwright
    elif action_type == BrowserActionTypes.CHECK:
        return f"create_check_action(pw_code={repr(action.data['pw_code'])})"
    elif action_type == BrowserActionTypes.SELECT_OPTION:
        return f"create_select_option_action(pw_code={repr(action.data['pw_code'])})"
    elif action_type == BrowserActionTypes.STOP:
        return f'create_stop_action({repr(action.data["answer"])})'
    else:
        raise ValueError(f"Invalid action type: {action.data['action_type']}")


class BrowserActionTypes(IntEnum):
    """Valid action types for browser env."""

    NONE = 0
    # mouse wheel and keyboard, universal across all action spaces
    SCROLL = 1
    KEY_PRESS = 2

    # low level mouse and keyboard actions
    MOUSE_CLICK = 3
    KEYBOARD_TYPE = 4
    MOUSE_HOVER = 5

    # mid level mouse and keyboard actions
    CLICK = 6
    TYPE = 7
    HOVER = 8

    # page level actions, universal across all action spaces
    PAGE_FOCUS = 9
    NEW_TAB = 10
    GO_BACK = 11
    GO_FORWARD = 12
    GOTO_URL = 13
    PAGE_CLOSE = 14

    # high-leval actions that playwright support
    CHECK = 15
    SELECT_OPTION = 16

    STOP = 17

    def __str__(self) -> str:
        return f"ACTION_TYPES.{self.name}"


@beartype
def is_equivalent(action: Action, other_action: Action) -> bool:
    """Return True iff two actions are equal."""
    if isinstance(action, BrowserAction) and isinstance(other_action, BrowserAction):
        return browser_is_equivalent(action, other_action)
    elif isinstance(action, BashAction) and isinstance(other_action, BashAction):
        return bash_is_equivalent(action, other_action)
    # We'll just say that two parsing errors are always equivalent
    elif isinstance(action, ParsingErrorAction) and isinstance(other_action, ParsingErrorAction):
        return True
    else:
        if isinstance(action, type(other_action)) and isinstance(other_action, type(action)):
            raise ValueError(
                f"didn't expect two actions to be of the same type here: {type(action)}"
            )
        # types don't match up
        return False


@beartype
def bash_is_equivalent(action: BashAction, other_action: BashAction) -> bool:
    if isinstance(action, BashStopAction) and isinstance(other_action, BashStopAction):
        return action.answer == other_action.answer
    elif isinstance(action, BashCommandAction) and isinstance(other_action, BashCommandAction):
        # Note: this could miss some equivalences if the command is formatted differently
        return action.command == other_action.command
    else:
        return True


@beartype
def browser_is_equivalent(a_action: BrowserAction, b_action: BrowserAction) -> bool:
    """Return True if two actions are equal."""
    a, b = a_action.data, b_action.data
    a_action_type, b_action_type = a["action_type"], b["action_type"]
    if a_action_type != b_action_type:
        return False

    # used to be match statement
    if a_action_type == BrowserActionTypes.NONE:
        return True
    elif a_action_type == BrowserActionTypes.SCROLL:
        da = "up" if "up" in a["direction"] else "down"
        db = "up" if "up" in b["direction"] else "down"
        return da == db
    elif a_action_type == BrowserActionTypes.KEY_PRESS:
        return a["key_comb"] == b["key_comb"]
    elif a_action_type in [BrowserActionTypes.MOUSE_CLICK, BrowserActionTypes.MOUSE_HOVER]:
        return np.allclose(a["coords"], b["coords"])
    elif a_action_type == BrowserActionTypes.KEYBOARD_TYPE:
        return a["text"] == b["text"]
    elif a_action_type in [
        BrowserActionTypes.CLICK,
        BrowserActionTypes.HOVER,
        BrowserActionTypes.TYPE,
    ]:  # TODO: can be further optimized
        if a["element_id"] and b["element_id"]:
            return a["element_id"] == b["element_id"]
        elif a["element_role"] and b["element_role"]:
            return a["element_role"] == b["element_role"] and a["element_name"] == b["element_name"]
        elif a["pw_code"] and b["pw_code"]:
            return a["pw_code"] == b["pw_code"]
        else:
            return False
    elif a_action_type == BrowserActionTypes.PAGE_FOCUS:
        return a["page_number"] == b["page_number"]
    elif a_action_type == BrowserActionTypes.NEW_TAB:
        return True
    elif a_action_type == BrowserActionTypes.GO_BACK:
        return True
    elif a_action_type == BrowserActionTypes.GO_FORWARD:
        return True
    elif a_action_type == BrowserActionTypes.GOTO_URL:
        return a["url"] == b["url"]
    elif a_action_type == BrowserActionTypes.PAGE_CLOSE:
        return True
    elif a_action_type in [BrowserActionTypes.CHECK, BrowserActionTypes.SELECT_OPTION]:
        return a["pw_code"] == b["pw_code"]
    elif a_action_type == BrowserActionTypes.STOP:
        return a["answer"] == b["answer"]
    else:
        raise ValueError(f"Unknown action type: {a['action_type']}")


_key2id: dict[str, int] = {
    key: i for i, key in enumerate(chain(SPECIAL_KEYS, ASCII_CHARSET, FREQ_UNICODE_CHARSET, ["\n"]))
}
_id2key: list[str] = sorted(_key2id, key=_key2id.get)  # type: ignore[arg-type]
_role2id: dict[RolesType, int] = {
    cast(RolesType, role): i for i, role in enumerate(chain(ROLES, SPECIAL_LOCATORS))
}
_id2role: list[RolesType] = sorted(_role2id, key=_role2id.get)  # type: ignore[arg-type]


@beartype
def _keys2ids(keys: Union[list[Union[int, str]], str]) -> list[int]:
    return list(
        map(
            lambda key: _key2id[str(key)] if is_bearable(key, str) else int(key),
            keys,
        )
    )


def get_action_space() -> spaces.Dict:
    """Return the space of serialized actions."""
    space = spaces.Dict(
        {
            "action_type": spaces.Discrete(len(BrowserActionTypes)),
            # coords (left, top) is used for COORD_CLICK
            "coords": spaces.Box(
                np.array([0.0, 0.0], dtype=np.float32),
                np.array([1.0, 1.0], dtype=np.float32),
            ),
            # element role is used for FOCUS_AND_CLICK and FOCUS_AND_TYPE
            "element_role": spaces.Discrete(len(ROLES) + len(SPECIAL_LOCATORS)),
            # element name is used with element role
            "element_name": spaces.Text(TEXT_MAX_LENGTH),
            "element_id": spaces.Text(TEXT_MAX_LENGTH),
            # text is only used for TYPE and FOCUS_AND_TYPE
            "text": spaces.MultiDiscrete(
                [len(ASCII_CHARSET) + len(SPECIAL_KEYS) + len(FREQ_UNICODE_CHARSET)]
                * TYPING_MAX_LENGTH
            ),
            "page_number": spaces.Discrete(MAX_PAGE_NUMBER),
            "url": spaces.Text(URL_MAX_LENGTH),
            "nth": spaces.Discrete(MAX_ELEMENT_INDEX_IN_VIEWPORT),
            "key_comb": spaces.Text(MAX_VANILLA_STR_LENGTH),
            "direction": spaces.Text(MAX_VANILLA_STR_LENGTH),
            "pw_code": spaces.Text(MAX_VANILLA_STR_LENGTH),
            "answer": spaces.Text(MAX_ANSWER_LENGTH),
        }
    )
    return space


def create_random_action() -> BrowserAction:
    """Return a random action."""
    return BrowserAction(
        is_stop=False,
        raw_prediction="",
        parsed_prediction="",
        data={
            "action_type": np.random.randint(len(BrowserActionTypes)),
            "coords": np.random.rand(2).astype(np.float32),
            "element_role": np.random.randint(len(ROLES) + len(SPECIAL_LOCATORS)),
            "element_name": "".join(
                random.choices(ASCII_CHARSET, k=np.random.randint(TEXT_MAX_LENGTH))
            ),
            "text": list(
                random.choices(
                    list(range(len(ASCII_CHARSET))),
                    k=np.random.randint(TYPING_MAX_LENGTH),
                )
            ),
            "page_number": np.random.randint(MAX_PAGE_NUMBER),
            "url": "".join(random.choices(ASCII_CHARSET, k=np.random.randint(URL_MAX_LENGTH))),
            "nth": np.random.randint(MAX_ELEMENT_INDEX_IN_VIEWPORT),
            "element_id": str(np.random.randint(MAX_ELEMENT_ID)),
            "key_comb": "+".join(random.choices(SPECIAL_KEYS, k=np.random.randint(3))),
            "direction": random.choice(["up", "down"]),
            "pw_code": "".join(
                random.choices(
                    string.ascii_uppercase + string.digits,
                    k=np.random.randint(MAX_VANILLA_STR_LENGTH),
                )
            ),
            "answer": str(np.random.randint(MAX_ANSWER_LENGTH)),
            "raw_prediction": str(np.random.randint(MAX_ANSWER_LENGTH)),
        },
    )


@beartype
def create_none_action() -> BrowserAction:
    """Return a valid action object that does nothing."""
    return BrowserAction(
        is_stop=False,
        raw_prediction="",
        parsed_prediction="",
        data={
            "action_type": BrowserActionTypes.NONE,
            "coords": np.zeros(2, dtype=np.float32),
            "element_role": 0,
            "element_name": "",
            "text": [],
            "page_number": 0,
            "url": "",
            "nth": 0,
            "pw_code": "",  # str that requires further processing
            "element_id": "",
            "key_comb": "",
            "direction": "",
            "answer": "",
            "raw_prediction": "",
        },
    )


@beartype
def create_stop_action(answer: str) -> BrowserAction:
    action = create_none_action()
    action.is_stop = True
    action.data.update({"action_type": BrowserActionTypes.STOP, "answer": answer})
    return action


@beartype
def create_scroll_action(direction: str) -> BrowserAction:
    """Return the playwright action"""
    assert direction in ["up", "down"]
    action = create_none_action()
    action.data.update(
        {
            "action_type": BrowserActionTypes.SCROLL,
            "direction": direction,
        }
    )
    return action


@beartype
def create_mouse_hover_action(
    left: Optional[float] = None, top: Optional[float] = None
) -> BrowserAction:
    """Return a valid action object with type COORD_CLICK."""
    action = create_none_action()
    action.data.update(
        {
            "action_type": BrowserActionTypes.MOUSE_HOVER,
            "coords": np.array([left, top], dtype=np.float32),
        }
    )
    return action


@beartype
def create_key_press_action(key_comb: str) -> BrowserAction:
    """Return the key press action"""

    def map_keys(key_comb: str) -> str:
        keys = key_comb.split("+")
        mapped_keys = []
        for key in keys:
            mapped_key = SPECIAL_KEY_MAPPINGS.get(key.lower(), key)
            mapped_keys.append(mapped_key)
        return "+".join(mapped_keys)

    action = create_none_action()
    mapped_key_comb = map_keys(key_comb)
    action.data.update(
        {
            "action_type": BrowserActionTypes.KEY_PRESS,
            "key_comb": mapped_key_comb,
        }
    )
    return action


@beartype
def create_page_focus_action(page_number: int) -> BrowserAction:
    """Return a valid action object with type PAGE_FOCUS."""
    action = create_none_action()
    action.data.update(
        {
            "action_type": BrowserActionTypes.PAGE_FOCUS,
            "page_number": page_number,
        }
    )
    return action


@beartype
def create_new_tab_action() -> BrowserAction:
    """Return a valid action object with type NEW_TAB."""
    action = create_none_action()
    action.data.update(
        {
            "action_type": BrowserActionTypes.NEW_TAB,
        }
    )
    return action


@beartype
def create_go_back_action() -> BrowserAction:
    """Return a valid action object with type GO_BACK."""
    action = create_none_action()
    action.data.update(
        {
            "action_type": BrowserActionTypes.GO_BACK,
        }
    )
    return action


@beartype
def create_go_forward_action() -> BrowserAction:
    """Return a valid action object with type GO_FORWARD."""
    action = create_none_action()
    action.data.update(
        {
            "action_type": BrowserActionTypes.GO_FORWARD,
        }
    )
    return action


@beartype
def create_goto_url_action(url: str) -> BrowserAction:
    """Return a valid action object with type GOTO_URL."""
    action = create_none_action()
    action.data.update(
        {
            "action_type": BrowserActionTypes.GOTO_URL,
            "url": url,
        }
    )
    return action


@beartype
def create_page_close_action() -> BrowserAction:
    """Return a valid action object with type PAGE_CLOSE."""
    action = create_none_action()
    action.data.update(
        {
            "action_type": BrowserActionTypes.PAGE_CLOSE,
        }
    )
    return action


@beartype
def create_mouse_click_action(
    left: Optional[float] = None, top: Optional[float] = None
) -> BrowserAction:
    """Return a valid action object with type COORD_CLICK."""
    action = create_none_action()
    if left and top:
        action.data.update(
            {
                "action_type": BrowserActionTypes.MOUSE_CLICK,
                "coords": np.array([left, top], dtype=np.float32),
            }
        )
    elif (not left) and (not top):
        action.data.update(
            {
                "action_type": BrowserActionTypes.CLICK,
            }
        )
    else:
        raise ValueError("left and top must be both None or both not None")
    return action


@beartype
def create_keyboard_type_action(keys: Union[list[Union[int, str]], str]) -> BrowserAction:
    """Return a valid action object with type TYPE."""
    action = create_none_action()
    action.data.update(
        {
            "action_type": BrowserActionTypes.KEYBOARD_TYPE,
            "text": _keys2ids(keys),
        }
    )
    return action


@beartype
def create_click_action(
    element_id: str = "",
    element_role: RolesType = "link",
    element_name: str = "",
    pw_code: str = "",
    nth: int = 0,
) -> BrowserAction:
    action = create_none_action()
    action.data.update(
        {
            "action_type": BrowserActionTypes.CLICK,
            "element_id": element_id,
            "element_role": _role2id[element_role],
            "element_name": element_name,
            "nth": nth,
            "pw_code": pw_code,
        }
    )
    return action


@beartype
def create_hover_action(
    element_id: str = "",
    element_role: RolesType = "link",
    element_name: str = "",
    pw_code: str = "",
    nth: int = 0,
) -> BrowserAction:
    action = create_none_action()
    action.data.update(
        {
            "action_type": BrowserActionTypes.HOVER,
            "element_id": element_id,
            "element_role": _role2id[element_role],
            "element_name": element_name,
            "nth": nth,
            "pw_code": pw_code,
        }
    )
    return action


@beartype
def create_type_action(
    text: str,
    element_id: str = "",
    element_role: RolesType = "link",
    element_name: str = "",
    pw_code: str = "",
    nth: int = 0,
) -> BrowserAction:
    action = create_none_action()
    action.data.update(
        {
            "action_type": BrowserActionTypes.TYPE,
            "element_id": element_id,
            "element_role": _role2id[element_role],
            "element_name": element_name,
            "nth": nth,
            "text": _keys2ids(text),
            "pw_code": pw_code,
        }
    )
    return action


@beartype
def create_check_action(pw_code: str) -> BrowserAction:
    action = create_none_action()
    action.data.update(
        {
            "action_type": BrowserActionTypes.CHECK,
            "pw_code": pw_code,
        }
    )
    return action


@beartype
def create_select_option_action(
    pw_code: str,
) -> BrowserAction:
    action = create_none_action()
    action.data.update(
        {
            "action_type": BrowserActionTypes.SELECT_OPTION,
            "pw_code": pw_code,
        }
    )
    return action


@beartype
def create_focus_action(
    element_role: RolesType, element_name: str = "", nth: int = 0
) -> BrowserAction:
    """Return a valid action object with type CLICK.

    Keep compatible with the old version."""
    action = create_none_action()
    action.data.update(
        {
            "action_type": BrowserActionTypes.CLICK,
            "element_role": _role2id[element_role],
            "element_name": element_name,
            "nth": nth,
        }
    )
    return action


@beartype
def create_focus_and_click_action(
    element_role: RolesType, element_name: str = "", nth: int = 0
) -> BrowserAction:
    """Return a valid action object with type CLICK.

    Keep compatible with the old version."""

    action = create_none_action()
    action.data.update(
        {
            "action_type": BrowserActionTypes.CLICK,
            "element_role": _role2id[element_role],
            "element_name": element_name,
            "nth": nth,
        }
    )
    return action


@beartype
def create_focus_and_type_action(
    keys: Union[list[Union[int, str]], str],
    element_role: RolesType,
    element_name: str = "",
    nth: int = 0,
) -> BrowserAction:
    """Return a valid action object with type TYPE.

    Keep compatible with the old version."""
    action = create_none_action()
    action.data.update(
        {
            "action_type": BrowserActionTypes.TYPE,
            "element_role": _role2id[element_role],
            "element_name": element_name,
            "text": _keys2ids(keys),
            "nth": nth,
        }
    )
    return action


@beartype
def execute_scroll(direction: str, page: PageForwarder) -> None:
    # perform the action
    # code from natbot
    if direction == "up":
        page.evaluate(
            "(document.scrollingElement || document.body).scrollTop = (document.scrollingElement || document.body).scrollTop - window.innerHeight;"
        )
    elif direction == "down":
        page.evaluate(
            "(document.scrollingElement || document.body).scrollTop = (document.scrollingElement || document.body).scrollTop + window.innerHeight;"
        )


@beartype
def execute_key_press(key: str, page: PageForwarder) -> None:
    """Press a key."""
    if "Meta" in key and "Mac" not in page.evaluate("navigator.platform"):
        key = key.replace("Meta", "Control")
    page.keyboard.press(key)


@beartype
def execute_mouse_hover(left: float, top: float, page: PageForwarder) -> None:
    """Click at coordinates (left, top)."""
    viewport_size = page.viewport_size
    assert viewport_size
    page.mouse.move(left * viewport_size["width"], top * viewport_size["height"])


def execute_mouse_click(left: float, top: float, page: PageForwarder) -> None:
    """Click at coordinates (left, top)."""
    viewport_size = page.viewport_size
    assert viewport_size
    page.mouse.click(left * viewport_size["width"], top * viewport_size["height"])


@beartype
def execute_keyboard_type(text: str, page: PageForwarder) -> None:
    """Fill the focused element with text."""
    page.keyboard.type(text)


@beartype
def execute_click_current(page: PageForwarder) -> None:
    """Click at the current mouse position."""
    raise NotImplementedError("execute_click_current is not implemented in flask-playwright api")


@beartype
def execute_type(keys: list[int], page: PageForwarder) -> None:
    """Send keystrokes to the focused element."""
    text = "".join([_id2key[key] for key in keys])
    page.keyboard.type(text)


@beartype
def execute_focus(element_role: int, element_name: str, nth: int, page: PageForwarder) -> None:
    """Click the specified DOM element."""
    raise NotImplementedError("execute_focus is not implemented in flask-playwright api")


@beartype
def locate(locator_calls: list[ParsedPlaywrightCode], page: PageForwarder) -> Locator:
    locator = page
    for call in locator_calls:
        function_name = call["function_name"]
        arguments = call["arguments"]
        keywords = call["keywords"]
        locator = getattr(locator, function_name)(*arguments, **keywords)
    return locator  # type: ignore[return-value]


@beartype
def execute_playwright_click(
    locator_code: list[ParsedPlaywrightCode],
    page: PageForwarder,
    pw_action_args: list[str] = [],
    pw_action_kwargs: dict[str, Any] = {},
) -> None:
    locator = locate(locator_code, page)

    # perform the action
    locator.click(*pw_action_args, **pw_action_kwargs)


@beartype
def execute_playwright_hover(locator_code: list[ParsedPlaywrightCode], page: PageForwarder) -> None:
    locator = locate(locator_code, page)

    # perform the action
    locator.hover()


@beartype
def execute_playwright_type(
    text: str,
    locator_code: list[ParsedPlaywrightCode],
    page: PageForwarder,
    pw_action_args: list[str] = [],
    pw_action_kwargs: dict[str, Any] = {},
) -> None:
    locator = locate(locator_code, page)
    # perform the action
    pw_action_args = [text] + pw_action_args  # text is the first argument
    locator.type(*pw_action_args, **pw_action_kwargs)


@beartype
def execute_playwright_select_option(
    locator_code: list[ParsedPlaywrightCode],
    page: PageForwarder,
    pw_action_args: list[str] = [],
    pw_action_kwargs: dict[str, Any] = {},
) -> None:
    locator = locate(locator_code, page)
    # perform the action
    locator.select_option(*pw_action_args, **pw_action_kwargs)


@beartype
def execute_playwright_check(locator_code: list[ParsedPlaywrightCode], page: PageForwarder) -> None:
    locator = locate(locator_code, page)
    # perform the action
    locator.check()


@beartype
def execute_action(
    action: BrowserAction,
    page: PageForwarder,
    obseration_processor: ObservationProcessor,
) -> None:
    """Execute the action on the ChromeDriver."""
    action_type = action.data["action_type"]
    # used to be match statement
    if action_type == BrowserActionTypes.NONE:
        pass
    # adding this to avoid errors
    elif action_type == BrowserActionTypes.STOP:
        pass

    elif action_type == BrowserActionTypes.SCROLL:
        direction = "up" if "up" in action.data["direction"] else "down"
        execute_scroll(direction, page)
    elif action_type == BrowserActionTypes.KEY_PRESS:
        keys = action.data["key_comb"]
        execute_key_press(keys, page)

    elif action_type == BrowserActionTypes.MOUSE_CLICK:
        execute_mouse_click(action.data["coords"][0], action.data["coords"][1], page)
    elif action_type == BrowserActionTypes.MOUSE_HOVER:
        execute_mouse_hover(action.data["coords"][0], action.data["coords"][1], page)
    elif action_type == BrowserActionTypes.KEYBOARD_TYPE:
        execute_type(action.data["text"], page)

    elif action_type == BrowserActionTypes.CLICK:
        # check each kind of locator in order
        # TODO[shuyanzh]: order is temp now
        if action.data["element_id"]:
            element_id = action.data["element_id"]
            element_center = obseration_processor.get_element_center(element_id)  # type: ignore[attr-defined]
            execute_mouse_click(element_center[0], element_center[1], page)
        elif action.data["element_role"] and action.data["element_name"]:
            raise NotImplementedError("Can't do locators with flask-playwright api yet")
        elif action.data["pw_code"]:
            parsed_code = parse_playwright_code(action.data["pw_code"])
            locator_code = parsed_code[:-1]
            # [shuyanzh], don't support action args and kwargs now
            execute_playwright_click(locator_code=locator_code, page=page)
        else:
            raise ValueError("No proper locator found for click action")
    elif action_type == BrowserActionTypes.HOVER:
        if action.data["element_id"]:
            element_id = action.data["element_id"]
            element_center = obseration_processor.get_element_center(element_id)  # type: ignore[attr-defined]
            execute_mouse_hover(element_center[0], element_center[1], page)
        elif action.data["element_role"] and action.data["element_name"]:
            element_role = int(action.data["element_role"])
            element_name = action.data["element_name"]
            nth = action.data["nth"]
            execute_focus(element_role, element_name, nth, page)
        elif action.data["pw_code"]:
            parsed_code = parse_playwright_code(action.data["pw_code"])
            locator_code = parsed_code[:-1]
            # [shuyanzh], don't support action args and kwargs now
            execute_playwright_hover(locator_code=locator_code, page=page)
        else:
            raise NotImplementedError("No proper locator found for hover action")
    elif action_type == BrowserActionTypes.TYPE:
        if action.data["element_id"]:
            element_id = action.data["element_id"]
            element_center = obseration_processor.get_element_center(element_id)  # type: ignore[attr-defined]
            execute_mouse_click(element_center[0], element_center[1], page)
            execute_type(action.data["text"], page)
        elif action.data["element_role"] and action.data["element_name"]:
            element_role = int(action.data["element_role"])
            element_name = action.data["element_name"]
            nth = action.data["nth"]
            execute_focus(element_role, element_name, nth, page)
            execute_type(action.data["text"], page)
        elif action.data["pw_code"]:
            parsed_code = parse_playwright_code(action.data["pw_code"])
            locator_code = parsed_code[:-1]
            text = parsed_code[-1]["arguments"][0]
            # [shuyanzh], don't support action args and kwargs now
            execute_playwright_type(text=text, locator_code=locator_code, page=page)
        else:
            raise NotImplementedError("No proper locator found for type action")
    elif action_type == BrowserActionTypes.GO_BACK:
        page.go_back()
    elif action_type == BrowserActionTypes.GO_FORWARD:
        page.go_forward()
    elif action_type == BrowserActionTypes.GOTO_URL:
        page.goto(action.data["url"])
    elif action_type == BrowserActionTypes.SELECT_OPTION:
        if action.data["pw_code"]:
            parsed_code = parse_playwright_code(action.data["pw_code"])
            locator_code = parsed_code[:-1]
            execute_playwright_select_option(locator_code, page)
        else:
            raise NotImplementedError("No proper locator found for select option action")
    elif action_type == BrowserActionTypes.CHECK:
        if action.data["pw_code"]:
            parsed_code = parse_playwright_code(action.data["pw_code"])
            locator_code = parsed_code[:-1]
            execute_playwright_check(locator_code, page)
        else:
            raise NotImplementedError("No proper locator found for select option action")

    else:
        raise ValueError(f"Unknown action type: {action_type}")


@beartype
def parse_playwright_code(code: str) -> list[ParsedPlaywrightCode]:
    # extract function calls
    if not code.startswith("page."):
        raise ValueError(f'Playwright action must start with "page.", but got {code}')

    regex = r"\.(?![^\(\)]*\))"
    chain = re.split(regex, code)[1:]

    parsed_chain = []

    for item in chain:
        tree = ast.parse(item)
        funcs = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                function_name = node.func.id  # type: ignore[attr-defined]
                arguments = [
                    ast.literal_eval(arg) if isinstance(arg, ast.Str) else arg for arg in node.args
                ]
                keywords = {str(kw.arg): ast.literal_eval(kw.value) for kw in node.keywords}
                funcs.append(
                    ParsedPlaywrightCode(
                        {
                            "function_name": function_name,
                            "arguments": arguments,  # type: ignore (seems to work fine)
                            "keywords": keywords,
                        }
                    )
                )

        if len(funcs) != 1:
            raise ValueError(f"Fail to parse {item} in {code}")

        if funcs[0]["function_name"] not in PLAYWRIGHT_LOCATORS + PLAYWRIGHT_ACTIONS:
            raise ValueError(
                f"Invalid playwright code {item}, ",
                f"the function needs to be one of {PLAYWRIGHT_LOCATORS + PLAYWRIGHT_ACTIONS}",
            )

        parsed_chain.append(funcs[0])

    last_action = parsed_chain[-1]
    if last_action.data["function_name"] not in PLAYWRIGHT_ACTIONS:
        raise ValueError(
            f"Invalid playwright action {last_action},",
            f"the action needs to be one of {PLAYWRIGHT_ACTIONS}",
        )

    return parsed_chain


@beartype
class ActionParsingError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


@beartype
def create_playwright_action(playwright_code: str) -> BrowserAction:
    """Main function to return individual playwright action"""
    # get the last action
    regex = r"\.(?![^\(\)]*\))"
    action = re.split(regex, playwright_code)[-1].split("(")[0]
    # used to be match statement
    if action == "press":
        p = r'press\((?:"|\')(.+?)(?:"|\')\)'
        match = re.search(p, playwright_code)
        if not match:
            raise ActionParsingError(
                "Invalid press action, required to be page.press(KEY_COMB_STR)"
            )
        key_comb = match.group(1)
        return create_key_press_action(key_comb=key_comb)
    elif action == "scroll":
        direction = "up" if "up" in playwright_code else "down"
        return create_scroll_action(direction=direction)
    elif action == "click":
        return create_click_action(pw_code=playwright_code)
    elif action == "hover":
        return create_hover_action(pw_code=playwright_code)
    elif action in ["type", "fill"]:
        p = r'type|fill\((?:"|\')(.+?)(?:"|\')\)'
        match = re.search(p, playwright_code)
        if not match:
            raise ActionParsingError("Invalid type/fill action, required to be page.type(TEXT)")
        text = match.group(1)
        return create_type_action(text=text, pw_code=playwright_code)
    elif action == "select_option":
        return create_select_option_action(pw_code=playwright_code)
    elif action == "check":
        return create_check_action(pw_code=playwright_code)
    elif action == "goto":
        p = r'goto\((?:"|\')(.+?)(?:"|\')\)'
        match = re.search(p, playwright_code)
        if not match:
            raise ActionParsingError("Invalid goto action, required to be page.goto(URL_STR)")
        url = match.group(1)
        return create_goto_url_action(url)
    elif action == "page_focus":
        # get the page number
        p = r"page_focus\((\d+)\)"
        match = re.search(p, playwright_code)
        if not match:
            raise ActionParsingError("page focus requires a page number")
        page_num = int(match.group(1))
        return create_page_focus_action(page_num)
    elif action == "new_tab":
        return create_new_tab_action()
    elif action == "go_back":
        return create_go_back_action()
    elif action == "go_forward":
        return create_go_forward_action()
    elif action == "page_close":
        return create_page_close_action()
    elif action == "stop":  # page.stop(answer)
        p = r'stop\(?"(.+)?"\)'
        match = re.search(p, playwright_code)
        if not match:
            answer = ""
        else:
            answer = match.group(1)
        return create_stop_action(answer)

    raise ActionParsingError(f"Unknown playwright action {action}")


@beartype
def create_id_based_action(action_str: str) -> BrowserAction:
    """Main function to return individual id based action"""
    action_str = action_str.strip()
    action = (
        action_str.split("[")[0].strip() if "[" in action_str else action_str.split()[0].strip()
    )
    # used to be match statement
    if action == "click":
        match = re.search(r"click ?\[(\d+)\]", action_str)
        if not match:
            raise ActionParsingError(f"Invalid click action {action_str}")
        element_id = match.group(1)
        return create_click_action(element_id=element_id)
    elif action == "hover":
        match = re.search(r"hover ?\[(\d+)\]", action_str)
        if not match:
            raise ActionParsingError(f"Invalid hover action {action_str}")
        element_id = match.group(1)
        return create_hover_action(element_id=element_id)
    elif action == "type":
        # add default enter flag
        if not (action_str.endswith("[0]") or action_str.endswith("[1]")):
            action_str += " [1]"

        match = re.search(r"type ?\[(\d+)\] ?\[(.+)\] ?\[(\d+)\]", action_str)
        if not match:
            raise ActionParsingError(f"Invalid type action {action_str}")
        element_id, text, enter_flag = (
            match.group(1),
            match.group(2),
            match.group(3),
        )
        if enter_flag == "1":
            text += "\n"
        return create_type_action(text=text, element_id=element_id)
    elif action == "press":
        match = re.search(r"press ?\[(.+)\]", action_str)
        if not match:
            raise ActionParsingError(f"Invalid press action {action_str}")
        key_comb = match.group(1)
        return create_key_press_action(key_comb=key_comb)
    elif action == "scroll":
        # up or down
        match = re.search(r"scroll ?\[?(up|down)\]?", action_str)
        if not match:
            raise ActionParsingError(f"Invalid scroll action {action_str}")
        direction = match.group(1)
        return create_scroll_action(direction=direction)
    elif action == "goto":
        match = re.search(r"goto ?\[(.+)\]", action_str)
        if not match:
            raise ActionParsingError(f"Invalid goto action {action_str}")
        url = match.group(1)
        return create_goto_url_action(url=url)
    elif action == "new_tab":
        return create_new_tab_action()
    elif action == "go_back":
        return create_go_back_action()
    elif action == "go_forward":
        return create_go_forward_action()
    elif action == "tab_focus":
        match = re.search(r"tab_focus ?\[(\d+)\]", action_str)
        if not match:
            raise ActionParsingError(f"Invalid tab_focus action {action_str}")
        page_number = int(match.group(1))
        return create_page_focus_action(page_number)
    elif action == "close_tab":
        return create_page_close_action()
    elif action == "stop":  # stop answer
        match = re.search(r"stop ?\[(.+)\]", action_str)
        if not match:  # some tasks don't require an answer
            answer = ""
        else:
            answer = match.group(1)
        return create_stop_action(answer)
    else:
        raise ActionParsingError(f"Invalid action {action_str}")
