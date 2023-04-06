from evals.api import OpenAIChatCompletionFn, OpenAICompletionFn
from evals.completion_fns import MyCompletionFn
from evals.registry import Registry


registry = Registry()
registry.api_model_ids = ["ada"]

def test_make_completion_fn_openai():
    assert isinstance(registry.make_completion_fn("ada"), OpenAICompletionFn)

def test_make_completion_fn_openai_chat():
    assert isinstance(registry.make_completion_fn("gpt-3.5-turbo"), OpenAIChatCompletionFn)

def test_make_completion_fn_class():
    assert isinstance(registry.make_completion_fn("evals.api:OpenAICompletionFn"), OpenAICompletionFn)

def test_make_completion_fn_custom_class():
    assert isinstance(registry.make_completion_fn("evals.completion_fns:MyCompletionFn"), MyCompletionFn)

