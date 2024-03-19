import json
import logging
import random
import re
from math import sqrt
from typing import Callable, Dict, List, Set, Union

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

ANSWER_PATTERN = r"\[ANSWER:? .+?\]"
VIEW_PATTERN = r"\[VIEW:? .+?\]"
PROMPTS = {
    "retrieval_instructions": (
        "This is an open-book test."
        " If you are unsure about a question or unable to answer, you may read through a number of"
        " knowledge sources that are made available to you in the form of text files."
        " To open a file, output [VIEW X] where X is the name of the source you would like."
        " For example, to view a file called help.jsonl, output [VIEW help.jsonl]."
        " When you first open a file I will tell you its table of contents."
        " You may view a particular section by outputting [VIEW X Y] where X is the name of"
        " the file and Y is the name of the section you would like to read."
        " The section titles won't necessarily match your exact task."
        " You should explore them for any useful information, such as key words or phrases,"
        " before giving your answer. You should always try to provide an answer,"
        " even if you are not confident. Failure to provide an answer is penalised"
        " more strongly than incorrect answers."
        "\nHere are the sources available to you: {list_of_files}."
    ),
    "intermediate_prompt": "You've already viewed the following files and sections: {sections}.\nYou can view another file or section by outputting [VIEW X] or [VIEW X Y], or you can answer the question by outputting [ANSWER X].",
    "present_content": "You asked to view file {file}, section {section}. Here is the content: {content}",
    "wrong_file": "You tried to view {file}, which does not exist in the knowledge base. Choose another file from {knowledge_base}.",
    "wrong_section": "You tried to view section {section} in file {file}, which does not exist. The table of contents for that file contains: {table_of_contents}.",
}

logger = logging.getLogger(__name__)


def answer_detected(output: str) -> bool:
    return len(re.findall(ANSWER_PATTERN, output)) > 0


def view_instruction_detected(output: str) -> bool:
    return len(re.findall(VIEW_PATTERN, output)) > 0


def process_answer(output: str) -> str:
    """Extracts the answer from model output.
    The answer looks like [ANSWER X], where X is the answer.

    Args:
        output (str): model output

    Returns:
        str: answer provided by the model
    """
    maybe_multiple_answers = re.findall(ANSWER_PATTERN, output)

    # Sanity check – this should never happen.
    assert len(maybe_multiple_answers) > 0, f"No answer detected in {output}."

    if len(maybe_multiple_answers) > 1:
        logger.debug(
            f"Multiple answers detected, using only the final answer: {maybe_multiple_answers}"
        )

    final_answer_instruction = maybe_multiple_answers[-1]
    final_answer = " ".join(final_answer_instruction.split(" ")[1:])[:-1]

    return final_answer


def process_view_instruction(output: str) -> Union[tuple[str, str], tuple[str, None]]:
    """Extracts the target of a view instruction from model output.
    The view instruction looks like [VIEW X Y], where X is a file name and Y is a section name.
    This function extracts X and Y.

    Args:
        output (str): model output

    Returns:
        Union[tuple[str, str], tuple[str, None]]: tuple of file name and if applicable section name to view
    """
    maybe_multiple_views = re.findall(VIEW_PATTERN, output)

    # Sanity check – this should never happen.
    assert len(maybe_multiple_views) > 0, f"No view instruction detected in {output}."

    if len(maybe_multiple_views) > 1:
        logger.debug(
            f"Multiple view instructions detected, using only the final instruction: {maybe_multiple_views}"
        )

    final_view_instruction = maybe_multiple_views[-1][1:-1].split(" ")[1:]
    file = final_view_instruction[0].strip()

    section = (
        None if len(final_view_instruction) == 1 else " ".join(final_view_instruction[1:]).strip()
    )

    return (file, section)


def _get_average_metric(
    results: List[Dict[str, str]], metric_fn: Callable[List[Dict[str, str]], List[float]]
) -> float:
    total_metric = sum(metric_fn(results))
    num_total = len(results)
    if num_total == 0:
        return float("nan")
    else:
        return total_metric / num_total


def get_bootstrap_accuracy_std(results: List[Dict[str, str]], num_samples: int = 1000) -> float:
    results = [sample for sample in results if sample["question_type"] != "translation"]
    vals = [result["correct"] for result in results]
    return np.std([np.mean(random.sample(vals, len(vals) // 2)) for _ in range(1000)])


def render_intermediate_prompt(sections_viewed: Dict[str, Set]) -> str:
    return PROMPTS["intermediate_prompt"].format(
        sections=json.dumps(
            {k: list(v) for k, v in sections_viewed.items()}, indent=4
        )  # Cannot serialise sets directly.
    )


def get_question_type(question: str) -> str:
    return "translation" if question.strip().startswith("Translate") else "non-translation"


def get_average_bleu_score(results: List[Dict[str, str]]) -> float:
    results = [sample for sample in results if sample["question_type"] == "translation"]
    return _get_average_metric(
        results,
        lambda samples: [
            get_bleu_score(sample["expected"][0], sample["parsed_output"]) for sample in samples
        ],
    )


def get_bleu_score(expected: str, sampled: str) -> float:
    punctuation = r"[^\w\s]"

    return sentence_bleu(
        [re.sub(punctuation, "", expected).split()],
        re.sub(punctuation, "", sampled).split(),
        smoothing_function=SmoothingFunction().method1,
    )


def get_accuracy(results: List[Dict[str, str]]) -> float:
    results = [sample for sample in results if sample["question_type"] != "translation"]
    return _get_average_metric(
        results, lambda samples: [int(sample["correct"]) for sample in samples]
    )


def get_average_retrieval_calls(results: List[Dict[str, str]]) -> float:
    return _get_average_metric(
        results, lambda samples: [sample["total_retrieval_calls"] for sample in samples]
    )


def get_average_invalid_retrieval_calls(results: List[Dict[str, str]]) -> float:
    return _get_average_metric(
        results, lambda samples: [sample["invalid_retrieval_calls"] for sample in samples]
    )


def get_average_retrieval_precision(results: List[Dict[str, str]]) -> float:
    return _get_average_metric(
        results, lambda samples: [sample["lesson_retrieval_calls"] for sample in samples]
    )


def get_std_of_difference(baseline_std: float, retrieval_std: float) -> float:
    return sqrt(baseline_std**2 + retrieval_std**2)
