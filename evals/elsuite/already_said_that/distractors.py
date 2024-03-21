import os
from dataclasses import dataclass
from pathlib import Path

import evals


@dataclass
class DistractorSample:
    question: str
    ideal: str


VARIANTS = {
    "which-is-heavier",
    "ambiguous-sentences",
    "first-letters",
    "reverse-sort-words-eng",
    "distractorless",
}


def proc_which_is_heavier(samples) -> list[DistractorSample]:
    distractor_samples = []
    for sample in samples:
        # get rid of ' Answer Yes or No'
        question = sample["input"][1]["content"][:-17]
        ideal = sample["ideal"].lower()
        distractor_samples.append(DistractorSample(question, ideal))
    return distractor_samples


def proc_distractors_first_letters(samples) -> list[DistractorSample]:
    distractor_samples = []
    for sample in samples:
        question = sample["input"][1]["content"]
        ideal = sample["ideal"].lower()
        distractor_samples.append(DistractorSample(question, ideal))
    return distractor_samples


def proc_distractors_ambiguous_sentences(samples) -> list[DistractorSample]:
    distractor_samples = []
    for sample in samples:
        sample_content = sample["input"][1]["content"]
        question = f"{sample_content}"
        ideal = sample["ideal"].lower()
        distractor_samples.append(DistractorSample(question, ideal))
    return distractor_samples


def proc_distractors_reverse_sort_words_eng(samples) -> list[DistractorSample]:
    distractor_samples = []
    for sample in samples:
        # cut " (respond as concisely as possible and only include the comma-separated words in your response):"
        instruction = sample["input"][0]["content"][:-96]
        sample_content = sample["input"][1]["content"]
        question = f"{instruction}: {sample_content}"
        ideal = sample["ideal"].lower()
        distractor_samples.append(DistractorSample(question, ideal))
    return distractor_samples


variant_to_processor = {
    "which-is-heavier": proc_which_is_heavier,
    "first-letters": proc_distractors_first_letters,
    "ambiguous-sentences": proc_distractors_ambiguous_sentences,
    "reverse-sort-words-eng": proc_distractors_reverse_sort_words_eng,
}


def get_basic_distractor_example() -> DistractorSample:
    """
    An arbitrary distractor example used in the task description for the
    distractorless variant
    """
    return DistractorSample(question="What is the capital of Italy?", ideal="rome")


def get_distractors(variant: str) -> list[DistractorSample]:
    """
    Gets and optionally processes the corpus of distractor questions for variant
    """
    assert variant in VARIANTS, f"Invalid variant {variant}, expected one of {VARIANTS}"
    if variant == "distractorless":
        # single element will be pop()ed for the task description, leaving an empty list
        return [get_basic_distractor_example()]

    samples = get_samples(variant)

    process_variant_fn = variant_to_processor[variant]
    processed_samples = process_variant_fn(samples)

    return processed_samples


def get_samples(eval_name) -> list[dict]:
    """
    Gets the samples from the samples_jsonl associated with
    a given eval.

    Adapted from evals.eval.Eval.get_samples
    """
    registry = evals.registry.Registry()
    eval_spec = registry.get_eval(eval_name)
    samples_path = eval_spec.args["samples_jsonl"]
    registry_path = eval_spec.registry_path
    samples_full_path = get_full_path(samples_path, registry_path)
    return evals.data.get_jsonl(samples_full_path.as_posix())


def get_full_path(data_path, registry_path) -> Path:
    if os.path.isfile(data_path):
        return Path(data_path)

    return registry_path / "data" / data_path


def get_distractor_word(question: str) -> str:
    """
    Takes the last word of the question (stripped of punctuation and lower-cased)
    To be shown in the task description example
    """
    words = question.split()
    last_word = words[-1]
    last_word = last_word.strip(".,!?")
    return last_word.lower()


if __name__ == "__main__":
    # just for testing
    distractors = get_distractors("rectangles")
    print(distractors[0])
