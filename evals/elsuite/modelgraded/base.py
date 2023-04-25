import string
from typing import TYPE_CHECKING, Any, Optional, Union

from evals.elsuite.modelgraded.classify_utils import (
    ANSWER_PROMPTS,
    choice_to_str,
    format_classify,
    get_choice,
)
from evals.elsuite.utils import PromptFn, format_prompt
from evals.prompt.base import OpenAICreateChatPrompt, is_chat_prompt

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass


@dataclass
class ModelGradedSpec:
    prompt: Union[str, OpenAICreateChatPrompt]
    choice_strings: Union[list[str], str]
    input_outputs: dict[str, str]

    eval_type: Optional[str] = None
    format_type: str = "in_message"
    choice_scores: Optional[Union[dict[str, Union[float, int]], str]] = None
    multicomp_n: Optional[int] = None
    expand_args_dict: Optional[dict[str, dict[str, tuple[str]]]] = None
    completion_sample_templates: Optional[dict[str, str]] = None
    match_fn: str = "starts_or_endswith"

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

        if isinstance(self.prompt, str):
            self.prompt = [{"role": "user", "content": self.prompt}]
        assert is_chat_prompt(self.prompt)

        # 'input_outputs' is a dict that specifies the input and output keys in the sample
        # output key is the model's raw response to input key. These are used for filling 'prompt' template.
        assert isinstance(
            self.input_outputs, dict
        ), f"input_outputs must be a dict, not {type(self.input_outputs)}"

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

    def append_answer_prompt(
        self,
        eval_type: str,
        append_type: str = "as_content",
        prompt: Optional[OpenAICreateChatPrompt] = None,
    ):
        """Append answer prompt to prompt. Can only be called once."""
        assert self.eval_type is None, f"eval_type already set: {eval_type}"
        prompt = prompt or ANSWER_PROMPTS[eval_type]
        prompt = format_prompt(prompt, choices=choice_to_str(self.choice_strings))
        if append_type == "as_content":
            assert isinstance(prompt, str), f"prompt must be str, not {type(prompt)}"
            self.prompt[-1]["content"] += "\n\n" + prompt
        elif append_type == "as_message":
            assert is_chat_prompt(prompt), f"prompt must be chat prompt, not {prompt}"
            self.prompt += prompt
        else:
            raise ValueError(f"append_type must be 'as_content' or 'as_message', not {append_type}")
        self.eval_type = eval_type

    def fill_args(self, **kwargs: dict[str, str]) -> OpenAICreateChatPrompt:
        """Fill 'prompt' template with 'kwargs'."""
        self.prompt = format_prompt(self.prompt, allow_missing=True, **kwargs)
        return self.prompt

    def format(self, **format_kwargs: dict[str, OpenAICreateChatPrompt]) -> OpenAICreateChatPrompt:
        return format_classify(
            prompt=self.prompt,
            input_outputs=self.input_outputs,
            format_type=self.format_type,
            **format_kwargs,
        )

    def classify(self, prompt: OpenAICreateChatPrompt, **completion_kwargs: dict[str, Any]) -> str:
        evaluate = PromptFn(prompt, **completion_kwargs)
        evaluation, _ = evaluate(skip_format=True)
        return get_choice(evaluation, self.eval_type, self.match_fn, self.choice_strings)
