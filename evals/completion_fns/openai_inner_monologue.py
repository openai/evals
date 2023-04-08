"""
Extending OpenAI Models with Chain-of-Thought Inner Monologue
"""
from evals import OpenAIChatCompletionFn, OpenAICompletionFn
from evals.prompt.base import ChatCompletionPrompt, CompletionPrompt
from evals.record import record_sampling


class OpenAIInnerMonologueCompletionResult:
    def __init__(self, response) -> None:
        self.response = response

    def get_completions(self) -> list[str]:
        return [self.response.strip()]


class OpenAIInnerMonologueCompletionFn:
    def __init__(self, **kwargs) -> None:
        self.cot_template = "\nBefore answering, I will reason in a step-by-step manner as to get the right answer, then conclude with the answer."
        self.extract_answer_template = (
            "\nGiven the above reasoning, the answer in the format requested by the question is:"
        )
        self.openai_completion_fn = OpenAICompletionFn(**kwargs)

    def __call__(self, prompt, **kwargs) -> OpenAIInnerMonologueCompletionResult:
        # Ensure it is in string format
        prompt = CompletionPrompt(prompt).to_formatted_prompt()

        cot_prompt = prompt + self.cot_template
        answer = self.openai_completion_fn(prompt=cot_prompt, **kwargs).get_completions()[0]
        record_sampling(prompt=cot_prompt, sampled=answer)

        extraction_prompt = cot_prompt + answer + self.extract_answer_template
        extracted_answer = self.openai_completion_fn(
            prompt=extraction_prompt, **kwargs
        ).get_completions()[0]
        record_sampling(prompt=extraction_prompt, sampled=extracted_answer)

        return OpenAIInnerMonologueCompletionResult(extracted_answer)


class OpenAIInnerMonologueChatCompletionFn:
    def __init__(self, **kwargs) -> None:
        self.cot_template = "Before answering, I will reason in a step-by-step manner as to get the right answer, then conclude with the answer."
        self.extract_answer_template = (
            "Given the above reasoning, the answer in the format requested by the question is:"
        )
        self.openai_chat_completion_fn = OpenAIChatCompletionFn(**kwargs)

    def __call__(self, prompt, **kwargs) -> OpenAIInnerMonologueCompletionResult:
        # Ensure it is in Chat format
        prompt = ChatCompletionPrompt(prompt).to_formatted_prompt()

        cot_prompt = prompt + [{"role": "assistant", "content": self.cot_template}]
        answer = self.openai_chat_completion_fn(prompt=cot_prompt, **kwargs).get_completions()[0]
        record_sampling(prompt=cot_prompt, sampled=answer)

        extraction_prompt = cot_prompt + [
            {"role": "assistant", "content": answer},
            {"role": "assistant", "content": self.extract_answer_template},
        ]
        extracted_answer = self.openai_chat_completion_fn(
            prompt=extraction_prompt, **kwargs
        ).get_completions()[0]
        record_sampling(prompt=extraction_prompt, sampled=extracted_answer)

        return OpenAIInnerMonologueCompletionResult(extracted_answer)
