import copy
import re
import string
from collections import Counter, defaultdict
from typing import Optional, Union

from evals import CompletionFn
from evals.prompt.base import (
    OpenAICreateChatPrompt,
    OpenAICreatePrompt,
    Prompt,
    chat_prompt_to_text_prompt,
    is_chat_prompt,
)


def get_answer(text, answer_prompt, ignore_case=False):
    if ignore_case:
        idx = text.lower().rfind(answer_prompt.lower())
    else:
        idx = text.rfind(answer_prompt)

    if idx == -1:
        return None
    return text[idx:idx + len(answer_prompt)]


def get_consensus(answers):
    counts = defaultdict(int)
    for answer in answers:
        counts[answer] += 1
    counts[None] = 0
    return max(counts, key=counts.get)


def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def fuzzy_match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)

    if s1 == "" or s2 == "":
        return s1 == s2

    return s1 in s2 or s2 in s1


def get_scores_from_text(text: str) -> dict:
    pattern = r"## (.+?)\n.+?(\d)/5"
    matches = re.findall(pattern, text, re.DOTALL)
    return {k: int(v) for k, v in dict(matches).items()}


def get_yesno_from_text(text: str) -> dict:
    pattern = r"## (.+?)\n.+?([yn])"
    matches = re.findall(pattern, text, re.DOTALL)
    return {k: v for k, v in dict(matches).items()}


def get_letter_from_data(data: str) -> str:
    last_y = (data.rfind("y"), "y")
    last_n = (data.rfind("n"), "n")
    char = max(last_y, last_n)[1]
    return char


def f1_score(prediction: str, answers: list[str]) -> float:
    def _f1_score(prediction: str, ground_truth: str):
        prediction_tokens = normalize(prediction).split()
        ground_truth_tokens = normalize(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    return max([_f1_score(prediction, answer) for answer in answers])


def scrub_formatting_from_prompt(prompt):
    scrubbed_prompt = copy.copy(prompt)

    if is_chat_prompt(prompt):
        for i, msg in enumerate(scrubbed_prompt):
            if "content" in msg:
                scrubbed_prompt[i]["content"] = msg["content"].replace("{", "{{").replace("}", "}}")
    else:
        scrubbed_prompt = scrubbed_prompt.replace("{", "{{").replace("}", "}}")
    return scrubbed_prompt


def format_necessary(template: str, allow_missing: bool = False, **kwargs: dict[str, str]) -> str:
    """Format a template string with only necessary kwargs."""
    keys = [k[1] for k in string.Formatter().parse(template) if k[1]]
    if allow_missing:
        assert (
            len([k for k in keys if k in kwargs]) > 0
        ), f"Required: {keys}, got: {sorted(kwargs)}, no inputs are used.\nTemplate:\n{template}"
        cur_keys = {k: kwargs.get(k, "{" + k + "}") for k in keys}
    else:
        assert all(
            k in kwargs for k in keys
        ), f"Required: {keys}, got: {sorted(kwargs)}.\nTemplate:\n{template}"
        cur_keys = {k: kwargs[k] for k in keys}
    return template.format(**cur_keys)


def format_prompt(
    prompt: OpenAICreatePrompt, allow_missing: bool = False, **kwargs: dict[str, str]
) -> OpenAICreatePrompt:
    """Format a prompt with only necessary kwargs."""
    # if any input kwargs is chat prompt, convert to text prompt
    kwargs = {
        k: chat_prompt_to_text_prompt(v, for_completion=False) if is_chat_prompt(v) else v
        for k, v in kwargs.items()
    }
    if is_chat_prompt(prompt):
        new_prompt = []
        for msg in prompt:
            formatted_msg = copy.copy(msg)
            if "content" in formatted_msg:
                formatted_msg["content"] = format_necessary(
                    formatted_msg["content"], allow_missing=allow_missing, **kwargs
                )
            new_prompt.append(formatted_msg)
        prompt = new_prompt
    else:
        # Prompt is a string
        prompt = format_necessary(prompt, allow_missing=allow_missing, **kwargs)
    return prompt


class PromptFn:
    """
    Wrap calls to a completion_fn with a prompt template with applicable keyword args.
    This will pass many args relevant to OpenAI Completion API, may be ignored by other completion_fn.
    """

    def __init__(
        self,
        prompt: Union[OpenAICreatePrompt, OpenAICreateChatPrompt, Prompt],
        completion_fn: CompletionFn,
        max_tokens: int,
        temperature: int = 0,
        n_samples: Optional[int] = None,
        completion_kwargs: Optional[dict] = {},
    ):
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.completion_fn = completion_fn
        self.temperature = temperature
        self.completion_kwargs = completion_kwargs
        self.n_samples = n_samples

    def __call__(self, **kwargs):
        # if any input kwargs is chat prompt, convert to text prompt
        kwargs = {
            k: chat_prompt_to_text_prompt(v, for_completion=False) if is_chat_prompt(v) else v
            for k, v in kwargs.items()
        }
        if is_chat_prompt(self.prompt):
            prompt = []
            for msg in self.prompt:
                formatted_msg = copy.copy(msg)
                if "content" in formatted_msg:
                    formatted_msg["content"] = format_necessary(formatted_msg["content"], **kwargs)
                prompt.append(formatted_msg)
        else:
            # Prompt is a string
            prompt = format_necessary(self.prompt, **kwargs)

        result = self.completion_fn(
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            n=(1 if self.n_samples is None else self.n_samples),
            **self.completion_kwargs,
        )
        sampled = result.get_completions()[0]
        return sampled, prompt
