from dataclasses import dataclass
from io import BytesIO
from typing import Any, Optional, TypedDict

import numpy as np
import numpy.typing as npt
from beartype import beartype
from PIL import Image

from evals.elsuite.multistep_web_tasks.webarena.core.env import EnvOutput, Info, Observation


@dataclass
class DetachedPage:
    url: str
    content: str  # html


@beartype
def png_bytes_to_numpy(png: bytes) -> npt.NDArray[np.uint8]:
    """Convert png bytes to numpy array

    Example:

    >>> fig = go.Figure(go.Scatter(x=[1], y=[1]))
    >>> plt.imshow(png_bytes_to_numpy(fig.to_image('png')))
    """
    return np.array(Image.open(BytesIO(png)))


class AccessibilityTreeNode(TypedDict):
    nodeId: str
    ignored: bool
    role: dict[str, Any]
    chromeRole: dict[str, Any]
    name: dict[str, Any]
    properties: list[dict[str, Any]]
    childIds: list[str]
    parentId: str
    backendDOMNodeId: int
    frameId: str
    bound: Optional[list[float]]
    union_bound: Optional[list[float]]
    offsetrect_bound: Optional[list[float]]


class BrowserWindowConfig(TypedDict):
    win_upper_bound: float
    win_left_bound: float
    win_width: float
    win_height: float
    win_right_bound: float
    win_lower_bound: float
    device_pixel_ratio: float


@dataclass
class PageInfo(Info):
    page: DetachedPage
    observation_metadata: dict[str, Any]
    fail_error: str


class BrowserState(TypedDict):
    config: BrowserWindowConfig
    DOMTree: dict[str, Any]


AccessibilityTree = list[AccessibilityTreeNode]


@dataclass
class BrowserObservation(Observation):
    acctree: Optional[str]
    html: Optional[str]
    image: Optional[npt.NDArray[np.uint8]]

    @property
    def data(self):
        raise NotImplementedError()

    def __repr__(self):
        return f"{self.__class__.__name__}(data={self.data})"


@dataclass(repr=False)
class HtmlBrowserObservation(BrowserObservation):
    @property
    def data(self):
        return self.html


@dataclass(repr=False)
class AccTreeBrowserObservation(BrowserObservation):
    @property
    def data(self):
        return self.acctree


@dataclass(repr=False)
class ImageBrowserObservation(BrowserObservation):
    @property
    def data(self):
        return self.image


@dataclass
class BrowserEnvOutput(EnvOutput):
    observation: BrowserObservation
    reward: float
    done: bool
    truncated: bool = False
    info: PageInfo
