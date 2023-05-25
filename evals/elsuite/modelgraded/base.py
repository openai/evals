from typing import TYPE_CHECKING, Optional, Union

from evals.prompt.base import OpenAICreateChatPrompt

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass


@dataclass
class ModelGradedSpec:
    # must have
    prompt: Union[str, OpenAICreateChatPrompt]
    choice_strings: Union[list[str], str]
    input_outputs: dict[str, str]

    # optional
    eval_type: Optional[str] = None
    choice_scores: Optional[Union[dict[str, float], str]] = None
    output_template: Optional[str] = None

    # unused
    key: Optional[str] = None
    group: Optional[str] = None
