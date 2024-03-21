from typing import Union

from evals.elsuite.multistep_web_tasks.webarena.bash_env.bash_utils import (
    BashEnvOutput,
    BashObservation,
)
from evals.elsuite.multistep_web_tasks.webarena.browser_env.browser_utils import (
    BrowserEnvOutput,
    BrowserObservation,
)

BashBrowserObservation = Union[BashObservation, BrowserObservation]

BashBrowserEnvOutput = Union[BashEnvOutput, BrowserEnvOutput]
