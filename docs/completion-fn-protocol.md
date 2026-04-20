### The Completion Function Protocol

Here are the interfaces needed to implement the completion function protocol. Any implementation of this interface can be used inside `oaieval`.

Reference implementations:
- [OpenAICompletionFn](../evals/completion_fns/openai.py)
- [LangChainLLMCompletionFn](../evals/completion_fns/langchain_llm.py)

#### CompletionFn
Completion functions should implement the `CompletionFn` interface:
```python
class CompletionFn(Protocol):
    def __call__(
        self,
        prompt: Union[str, list[dict[str, str]]],
        **kwargs,
    ) -> CompletionResult:
```

We take a `prompt` representing a single sample from an eval. These prompts can be represented as either a text string or a list of messages in [OpenAI Chat format](https://platform.openai.com/docs/guides/chat/introduction). To work with the existing evals, Completion Function implementations would need to handle both types of inputs, but we provide helper functionality to convert Chat formatted messages into a text string if that is the preferred input for your program:
```python
from evals.prompt.base import CompletionPrompt

# chat_prompt: list[dict[str, str]] -> text_prompt: str
text_prompt = CompletionPrompt(chat_prompt).to_formatted_prompt()
```

#### CompletionResult
The completion function should return an object implementing the `CompletionResult` interface:
```python
class CompletionResult(ABC):
    @abstractmethod
    def get_completions(self) -> list[str]:
        pass
```
The `get_completions` method returns a list of string completions. Each element should be considered a unique completion (in most cases this will be a list of length 1).

#### Using your CompletionFn
This is all that's needed to implement a Completion function that works with our existing Evals, allowing you to more easily evaluate your end-to-end logic on tasks.

See [completion-fns.md](completion-fns.md) to find out how to register and use your completion function with `oaieval`.
