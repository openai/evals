"""
Extending Completion Functions with Chain-of-Thought
"""
from evals.api import CompletionFn, CompletionResult
from evals.prompt.base import ChatCompletionPrompt
from evals.record import record_sampling
from evals.registry import Registry

DEFAULT_COT_TEMPLATE = "\nBefore answering, reason in a step-by-step manner as to get the right answer, then conclude with the answer."
DEFAULT_EXTRACT_ANSWER_TEMPLATE = (
    "\nGiven the above reasoning, the answer in the format requested by the question is:"
)


class ChainOfThoughtCompletionResult(CompletionResult):
    def __init__(self, response) -> None:
        self.response = response

    def get_completions(self) -> list[str]:
        return [self.response.strip()]


class ChainOfThoughtCompletionFn(CompletionFn):
    def __init__(
        self,
        cot_template: str = DEFAULT_COT_TEMPLATE,
        extract_answer_template: str = DEFAULT_EXTRACT_ANSWER_TEMPLATE,
        cot_completion_fn: str = None,
        extract_completion_fn: str = None,
        registry: Registry = None,
        registry_path: str = None,
        **kwargs
    ) -> None:
        registry = Registry() if not registry else registry
        if registry_path:
            registry.add_registry_paths(registry_path)

        if extract_completion_fn is None:
            extract_completion_fn = cot_completion_fn

        # This model will use chain of thought to answer the question
        self.cot_template = cot_template
        self.cot_completion_fn_instance = registry.make_completion_fn(cot_completion_fn)

        # This model will extract the answer from the chain of thought
        self.extract_answer_template = extract_answer_template
        self.extract_completion_fn_instance = registry.make_completion_fn(extract_completion_fn)

    def __call__(self, prompt, **kwargs) -> ChainOfThoughtCompletionResult:
        # Ensure it is in string format
        prompt = ChatCompletionPrompt(prompt).to_formatted_prompt()

        cot_prompt = prompt + [{"role": "assistant", "content": self.cot_template}]
        answer = self.cot_completion_fn_instance(prompt=cot_prompt, **kwargs).get_completions()[0]
        record_sampling(prompt=cot_prompt, sampled=answer)

        extraction_prompt = cot_prompt + [
            {"role": "assistant", "content": answer},
            {"role": "assistant", "content": self.extract_answer_template},
        ]
        extracted_answer = self.extract_completion_fn_instance(
            prompt=extraction_prompt, **kwargs
        ).get_completions()[0]
        record_sampling(prompt=extraction_prompt, sampled=extracted_answer)

        return ChainOfThoughtCompletionResult(extracted_answer)
