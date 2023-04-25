import string
from typing import TYPE_CHECKING, Any, Optional, Union

from evals.elsuite.modelgraded.classify_utils import (
    INVALID_STR,
    append_answer_prompt,
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
    eval_type: str

    format_type: str = "in_message"
    choice_scores: Optional[Union[dict[str, Union[float, int]], str]] = None
    multicomp_n: Optional[int] = None
    completion_sample_templates: Optional[dict[str, str]] = None
    match_fn: str = "starts_or_endswith"
    append_answer_prompt: bool = False
    append_type: str = "as_content"
    args: Optional[dict[str, Any]] = None

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
            self.max_score = max(self.choice_scores.values())
            self.min_score = min(self.choice_scores.values())

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
        if self.multicomp_n and self.multicomp_n > 1:
            assert (
                self.completion_sample_templates
            ), "completion_sample_templates must be specified if multicomp_n > 1"

        if self.append_answer_prompt:
            self.prompt = append_answer_prompt(
                prompt=self.prompt,
                eval_type=self.eval_type,
                choice_strings=self.choice_strings,
                append_type=self.append_type,
            )

        if self.args:
            self.prompt = format_prompt(self.prompt, allow_missing=True, **self.args)

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
        choice = get_choice(evaluation, self.eval_type, self.match_fn, self.choice_strings)
        return choice, evaluation

    def score(self, choice: str):
        # assumption: each INVALID_STR contributes the lowest score
        if choice == INVALID_STR:
            return self.min_score
        return self.choice_scores[choice]
