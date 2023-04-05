from evals.registry import Registry


registry = Registry()
spec = registry.get_completion_fn("test-completion")
complete = registry.make_completion_fn(spec)

prompt = "how many fingers do you have?"
result = complete(None, prompt)
print(result)