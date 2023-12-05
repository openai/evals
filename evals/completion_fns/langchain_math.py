import os

from langchain.chains import LLMMathChain
from langchain.llms import OpenAI

from evals.api import CompletionFn, CompletionResult
from evals.prompt.base import CompletionPrompt
from evals.record import record_sampling


class LangChainCompletionResult(CompletionResult):
    def __init__(self, response) -> None:
        self.response = response

    def get_completions(self) -> list[str]:
        return [self.response.strip()]


class LangChainMathChainCompletionFn(CompletionFn):
    def __init__(self, **kwargs) -> None:
        llm = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), temperature=0)
        self.llm_math = LLMMathChain(llm=llm)

    def __call__(self, prompt, **kwargs) -> LangChainCompletionResult:

        prompt = CompletionPrompt(prompt).to_formatted_prompt()
        response = self.llm_math.run(prompt)
        # The LangChain response comes with `Answer: ` ahead of this, let's strip it out
        response = response.strip("Answer:").strip()
        record_sampling(prompt=prompt, sampled=response)
        return LangChainCompletionResult(response)
