import json
from typing import Any, Union

from evals.api import CompletionFn, CompletionResult
from evals.prompt.base import OpenAICreateChatPrompt
from evals.record import record_sampling


def _prompt_key(prompt: Any) -> str:
    """Return a stable key for string or structured prompts."""
    return json.dumps(prompt, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class PrecomputedCompletionResult(CompletionResult):
    def __init__(self, completion: str):
        self.completion = completion

    def get_completions(self) -> list[str]:
        return [self.completion]


class PrecomputedCompletionFn(CompletionFn):
    """Serve completions already stored alongside eval samples.

    Samples must contain an ``input`` field and a string field named by
    ``output_key`` (``output`` by default). Inputs are indexed once when the
    completion function is created, so calls during evaluation perform no
    network requests.
    """

    def __init__(self, samples: list[dict[str, Any]], output_key: str = "output"):
        self.output_key = output_key
        self._outputs: dict[str, str] = {}

        for index, sample in enumerate(samples):
            if not isinstance(sample, dict):
                raise ValueError(f"Precomputed sample #{index} must be a dict")
            if "input" not in sample:
                raise ValueError(f"Precomputed sample #{index} is missing 'input'")
            if output_key not in sample:
                raise ValueError(
                    f"Precomputed sample #{index} is missing '{output_key}'. "
                    f"Add the generated response to that field."
                )

            output = sample[output_key]
            if not isinstance(output, str):
                raise ValueError(
                    f"Precomputed sample #{index} field '{output_key}' must be a string"
                )

            key = _prompt_key(sample["input"])
            if key in self._outputs and self._outputs[key] != output:
                raise ValueError(
                    "Precomputed samples contain the same input with different outputs. "
                    "Inputs must uniquely identify their stored output."
                )
            self._outputs[key] = output

    def __call__(
        self,
        prompt: Union[str, OpenAICreateChatPrompt],
        **_kwargs: Any,
    ) -> PrecomputedCompletionResult:
        key = _prompt_key(prompt)
        if key not in self._outputs:
            raise KeyError(
                "No precomputed output matches this prompt. The built-in precomputed mode "
                "expects the eval to pass each sample's 'input' to the completion function unchanged."
            )

        completion = self._outputs[key]
        result = PrecomputedCompletionResult(completion)
        record_sampling(prompt=prompt, sampled=result.get_completions(), model="precomputed")
        return result
