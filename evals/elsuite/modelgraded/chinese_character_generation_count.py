import random
import textwrap
import evals
import evals.metrics
import re
from evals.api import *
import string


def count_zh_char(text):
    pattern = re.compile(r'[\u4e00-\u9fa5]') 
    results = re.findall(pattern, text)
    count = 0
    for result in results:
        if not is_chinese_punctuation(result):
            count += 1
    return count


def is_chinese_punctuation(char):
    punctuation = '，。！？；：‘’“”【】（）<>《》『』［］｛｝〔〕〈〉「」「」﹂。、'
    return char in punctuation


def count_en_words(text):
    # remove all symbols
    text = text.translate(str.maketrans('', '', string.punctuation))

    words = text.split() 
    count = 0
    for word in words:
        if word.isalpha(): 
            count += 1
    return count


def check_sampled_text_self(
    model_spec: ModelSpec,
    prompt: Union[OpenAICreatePrompt, OpenAICreateChatPrompt, Prompt],
    expected: Union[str, list[str], tuple[str]],
    *,
    options: Optional[list[str]] = None,
    separator: Callable[[str], bool] = None,
    lang: str = None,
) -> Optional[str]:
    """
    Generates a completion using the prompt, checks whether the completion is
        one of the expected completions, and then records the result.

    ARGS
    ====
    `model_spec`: See `completion_query`.
    `prompt`: See `completion_query`.
    `options`: The list of canonical options, defaults to `expected` if None.
        The completion will be converted to one of these options.
    `expected`: The desired completion or the list of desired completions.
    `separator`: A callable which check the character sampled after the option
        to see if it is a valid separator.

    RETURNS
    =======
    The option that was picked, i.e., matched the completion, or None.
    """
    if isinstance(expected, tuple):
        expected = list(expected)
    elif not isinstance(expected, list):
        expected = [expected]
    if options is None:
        options = expected

    result, actual_prompt, metadata = completion_query(
        prompt=prompt,
        temperature=0.0,
        model_spec=model_spec,
    )
    choice = result["choices"][0]

    sampled = choice["text"].strip() if model_spec.strip_completion else choice["text"]

    # Here we will count how many chinese characters / english words in the response.
    count = 0
    if lang == "zh":
        count = count_zh_char(sampled)
    elif lang == "en":
        count = count_en_words(sampled)

    sampled = str(count)
    # Count OK.

    picked = None
    for option in options:
        if not sampled.startswith(option):
            continue
        if (
            separator is not None
            and len(sampled) > len(option)
            and not separator(sampled[len(option)])
        ):
            continue
        picked = option
        break

    result = {
        "prompt": actual_prompt,
        "sampled": sampled,
        "options": options,
        "picked": picked,
    }
    match = picked in expected
    result["expected"] = expected
    result["match"] = match
    result["metadata"] = metadata
    record_sampling(**result)
    record_match(match, expected=expected, picked=picked, sampled=sampled)
    return picked


class ChineseCharacterGenerationCount(evals.Eval):
    """
    Give an instruction containing how many Chinese characters to generate, 
    and let model generate specific amount of chinese characters, 
    finally evaluate the performance.
    """
    def __init__(self, jsonl, **kwargs):
        super().__init__(**kwargs)
        self.jsonl = jsonl

    def run(self, recorder):
        """
        Called by the `oaieval` CLI to run the eval. The `eval_all_samples` method calls `eval_sample`.
        """
        samples = evals.get_jsonl(self.jsonl)
        self.eval_all_samples(recorder, samples)

        # Record overall metrics
        return {
            "accuracy": evals.metrics.get_accuracy(recorder.get_events("match")),
        }
    
    
    def eval_sample(self, test_sample, rng: random.Random):
        """
        Called by the `eval_all_samples` method to evaluate a single sample.

        ARGS
        ====
        `test_sample`: a line from the JSONL test file
        `rng`: should be used for any randomness that is needed during evaluation

        This method does the following:
        1. Generate a prompt that contains the task statement, a few examples, and the test question.
        2. Check if the model generates the correct answer.
        """

        prompt = [
            {"role": "system", "content": "Give an instruction containing how many Chinese characters to generate, you should generate specific amount of chinese characters."},
        ]

        prompt += [{"role": "user", "content": test_sample["problem"]}]

        print(test_sample)

        # Define a new 'check_sampled_text' function called 'check_sampled_text_self'
        check_sampled_text_self(self.model_spec, prompt, expected=test_sample["count"], lang=test_sample["lang"])


