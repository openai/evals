import string
from typing import TYPE_CHECKING, Optional, Union

from evals.elsuite.modelgraded.classify_utils import ANSWER_PROMPTS, choice_to_str, expand_args_dict
from evals.prompt.base import OpenAICreateChatPrompt

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass


@dataclass
class ModelGradedSpec:
    prompt: Union[str, OpenAICreateChatPrompt]
    choice_strings: Union[list[str], str]
    eval_type: str
    input_outputs: dict[str, str]

    choice_scores: Optional[Union[dict[str, Union[float, int]], str]] = None
    multicomp_n: Optional[int] = None
    append_answer_prompt: bool = False
    args: Optional[dict[str, dict[str, str]]] = None
    expand_args_dict: Optional[dict[str, dict[str, tuple[str]]]] = None
    completion_sample_templates: Optional[dict[str, str]] = None

    key: Optional[str] = None  # unused
    group: Optional[str] = None  # unused

    def __post_init__(self):
        # 'choice_strings' is a list of strings that specifies the possible choices
        if self.choice_strings == "from_n":
            self.choice_strings = [str(i + 1) for i in range(self.multicomp_n)]
        elif self.choice_strings == "from_n_abc":
            self.choice_strings = [string.ascii_lowercase[i % 26] for i in range(self.multicomp_n)]
        elif self.choice_strings == "from_n_ABC":
            self.choice_strings = [string.ascii_uppercase[i % 26] for i in range(self.multicomp_n)]
        # make sure each choice doesn't contain any punctuation
        for s in self.choice_strings:
            assert not any(c in s for c in string.punctuation), f"{s} contains punctuation"

        #  (optional) 'choice_scores' is a dict that specifies the score for each choice string
        # if 'choice_scores' is specified, 'scores/' are computed and added to metrics
        if self.choice_scores:
            if self.choice_scores == "from_strings":
                self.choice_scores = {c: float(c) for c in self.choice_strings}

        # 'prompt' is a string that specifies the model-graded evaluation
        assert isinstance(self.prompt, str), f"prompt must be a string, not {type(self.prompt)}"
        if self.append_answer_prompt:
            self.prompt += "\n\n" + ANSWER_PROMPTS[self.eval_type].format(
                choices=choice_to_str(self.choice_strings)
            )
        self.prompt = [{"role": "user", "content": self.prompt}]

        # 'input_outputs' is a dict that specifies the input and output keys in the sample
        # output key is the model's raw response to input key. These are used for filling 'prompt' template.
        assert isinstance(
            self.input_outputs, dict
        ), f"input_outputs must be a dict, not {type(self.input_outputs)}"

        # (optional) 'args' is a dict of dicts that specifies additional arguments for 'prompt'
        # each value in 'args' essentially defines a separate modelgraded classification eval and has own metrics!
        self.args = self.args or {}
        self.expanded_args_dict = expand_args_dict(self.args)

        # (optional) 'completion_sample_templates'
        # each key must be one of 'input_outputs'.values(). If 'multicomp_n' > 1, this template is filled 'multicomp_n' times
        # and the concatenated result is passed to 'prompt' template.
        self.completion_sample_templates = self.completion_sample_templates or {}
        assert all(
            k in self.input_outputs.values() for k in self.completion_sample_templates
        ), f"all {self.completion_sample_templates.keys()} must be in {self.input_outputs.values()}, "
        if self.multicomp_n > 1:
            assert (
                self.completion_sample_templates
            ), "completion_sample_templates must be specified if multicomp_n > 1"
