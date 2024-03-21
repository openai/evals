DEFAULT_TASK_DESCRIPTION = "Solve the given problem, writing your reasoning along the way."

DEFAULT_MISTAKE_MESSAGE = "There might be a mistake in your reasoning."

DEFAULT_FINAL_ANSWER_MESSAGE = (
    "Given this reasoning, write your final answer. Only write your final answer, and nothing else."
)

TASK_SPECIFIC_EXTRACTION_INFO = {
    "dyck_languages": "\n\nAnswer with just the end of the sequence, separated by spaces. Do not repeat the part of the sequence given in the question. Only write the sequence of symbols, nothing else.",
    "logical_deduction": "\n\nAnswer with the selected single letter indicating your answer, wrapped with parentheses. Do not write anything else.",
    "multistep_arithmetic": "\n\nAnswer with a single number.",
    "tracking_shuffled_objects": "\n\nAnswer with the selected single letter indicating your answer, wrapped with parentheses. Do not write anything else.",
    "word_sorting": "\n\nAnswer with the sorted words, each lower case and separated by a single space.",
}
