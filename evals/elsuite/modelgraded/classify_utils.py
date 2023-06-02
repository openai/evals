import logging
import string
from typing import Any, Callable, Iterable, Optional, Union

from evals import CompletionFn
from evals.elsuite.modelgraded.base import ModelGradedSpec
from evals.elsuite.utils import PromptFn, format_necessary, format_prompt
from evals.prompt.base import OpenAICreateChatPrompt, is_chat_prompt

INVALID_STR = "__invalid__"


ANSWER_PROMPTS = {
    # e.g. "Yes"
    "classify": "Answer the question by printing only a single choice from {choices} (without quotes or punctuation) corresponding to the correct answer with no other text.".strip(),
    # e.g. "Yes\n The reasons are: ..."
    "classify_cot": "First, answer by printing a single choice from {choices} (without quotes or punctuation) corresponding to the correct answer. Then, from the next line, explain your reasonings step by step.".strip(),
    # e.g. "Let's think step by step. ...\nYes"
    "cot_classify": """
First, write out in a step by step manner your reasoning to be sure that your conclusion is correct. Avoid simply stating the correct answer at the outset. Then print only a single choice from {choices} (without quotes or punctuation) on its own line corresponding to the correct answer. At the end, repeat just the answer by itself on a new line.

Reasoning:""".strip(),
    "cot_classify_jp": """
まず、一歩一歩あなたの推論を書き出してください。単に正しい答えを最初に述べることを避けてください。次に、{choices}（引用符や句読点なし）から正しい答えに対応する1つの選択肢を単独の行に書きだしてください。最後に、答えだけを新しい行に繰り返してください。

推論：
    """.strip(),
}
MATCH_FNS = {
    "include": lambda x, y: float(x in y),
    "exact": lambda x, y: float(x == y),
    "endswith": lambda x, y: x.endswith(y),
    "starts_or_endswith": lambda x, y: x.startswith(y) or x.endswith(y),
}


def get_choice_strings(choice_strings: Union[list[str], str], n: Optional[int] = None):
    # 'choice_strings' is a list of strings that specifies the possible choices
    if choice_strings == "from_n":
        choice_strings = [str(i + 1) for i in range(n)]
    elif choice_strings == "from_n_abc":
        choice_strings = [string.ascii_lowercase[i % 26] for i in range(n)]
    elif choice_strings == "from_n_ABC":
        choice_strings = [string.ascii_uppercase[i % 26] for i in range(n)]
    # make sure each choice doesn't contain any punctuation
    for s in choice_strings:
        assert not any(c in s for c in string.punctuation), f"{s} contains punctuation"
    return choice_strings


def classify(
    mg: ModelGradedSpec,
    completion_fn: CompletionFn,
    completion_kwargs: Optional[dict[str, Any]] = None,
    format_kwargs: Optional[dict[str, Any]] = None,
    eval_type: Optional[str] = None,
    n: Optional[int] = None,
    match_fn: str = "starts_or_endswith",
) -> str:
    completion_kwargs = completion_kwargs or {}
    format_kwargs = format_kwargs or {}

    # get choice strings
    choice_strings = get_choice_strings(mg.choice_strings, n=n)

    # append answer prompt
    prompt = mg.prompt
    if isinstance(prompt, str):
        prompt = [{"role": "user", "content": prompt}]
    if eval_type:
        prompt = append_answer_prompt(
            prompt=prompt,
            eval_type=eval_type,
            choice_strings=choice_strings,
        )

    evaluate = PromptFn(prompt, completion_fn=completion_fn, **completion_kwargs)
    evaluation, prompt = evaluate(n=n, **format_kwargs)
    choice = get_choice(evaluation, mg.eval_type or eval_type, match_fn, choice_strings)
    score = get_choice_score(choice, choice_strings, mg.choice_scores)
    return choice, dict(
        score=score,
        sampled=[evaluation],
        prompt=prompt,
        invalid_choice=choice == INVALID_STR,
    )


def get_choice_score(
    choice: str,
    choice_strings: Iterable[str],
    choice_scores: Optional[Union[dict[str, float], str]] = None,
) -> Optional[float]:
    if choice_scores is None:
        return None
    if choice_scores == "from_strings":
        choice_scores = {c: float(c) for c in choice_strings}
    # assumption: each INVALID_STR contributes the lowest score
    if choice == INVALID_STR:
        return min(choice_scores.values())
    return choice_scores[choice]


def choice_to_str(choice_strings: Iterable[str]) -> str:
    """Return a string of choices, e.g. '"Yes" or "No" or "Maybe"'."""
    return " or ".join(f'"{choice}"' for choice in choice_strings)


def get_choice(
    text: str, eval_type: str, match_fn: Union[str, Callable], choice_strings: Iterable[str]
) -> str:
    """Clean the answer string to a choice string to one of choice_strings. Return '__invalid__.' if no match."""
    if isinstance(match_fn, str):
        match_fn = MATCH_FNS[match_fn]
    lines = text.strip().split("\n")
    if eval_type.startswith("cot_classify"):
        lines = lines[::-1]  # reverse lines
    for line in lines:
        line = line.strip()
        line = "".join(c for c in line if c not in string.punctuation)
        if not line:
            continue
        for choice in choice_strings:
            if match_fn(line, choice):
                return choice
    logging.warn(f"Choices {choice_strings} not parsable for {eval_type}: {text}")
    return INVALID_STR


def append_answer_prompt(
    prompt: OpenAICreateChatPrompt,
    eval_type: str,
    append_type: str = "as_content",
    answer_prompt: Optional[OpenAICreateChatPrompt] = None,
    choice_strings: Optional[Iterable[str]] = None,
) -> OpenAICreateChatPrompt:
    """Append answer prompt to prompt."""
    answer_prompt = answer_prompt or ANSWER_PROMPTS[eval_type]
    answer_prompt = format_prompt(answer_prompt, choices=choice_to_str(choice_strings))
    if append_type == "as_content":
        assert isinstance(answer_prompt, str), f"prompt must be str, not {type(answer_prompt)}"
        prompt[-1]["content"] += "\n\n" + answer_prompt
    elif append_type == "as_message":
        assert is_chat_prompt(answer_prompt), f"prompt must be chat prompt, not {answer_prompt}"
        prompt += answer_prompt
    else:
        raise ValueError(f"append_type must be 'as_content' or 'as_message', not {append_type}")
    return prompt


def sample_and_concat_n_completions(
    completion_fns: list[CompletionFn],
    prompt: OpenAICreateChatPrompt,
    n: int,
    template_i: str,
    sample_kwargs: dict,
):
    assert template_i
    completion_i_s = []
    for i in range(n):
        if len(completion_fns) > 1:
            # use a separate model for each completion
            assert len(completion_fns) == n
            completion_fn = completion_fns[i]
        else:
            # use the single model for all completions
            completion_fn = completion_fns[0]
        get_input_completion = PromptFn(prompt, completion_fn=completion_fn, **sample_kwargs)
        completion_i, _ = get_input_completion()
        completion_i_s.append(completion_i)
    return concat_n_completions(completion_i_s, template_i=template_i)


def concat_n_completions(completions: Iterable[str], template_i: str) -> str:
    """Concatenate n completions into a single text string."""
    completion = ""
    for i, completion_i in enumerate(completions):
        completion += format_necessary(
            template_i,
            i=i + 1,
            i_abc=string.ascii_lowercase[i % 26],
            i_ABC=string.ascii_uppercase[i % 26],
            output=completion_i,
            n=len(completions),
        )
    return completion.strip()
