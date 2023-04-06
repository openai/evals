from evals.api import CompletionFn

class MyCompletionResult():
    def __init__(self, raw_data) -> None:
        self.raw_data = raw_data

    def get_completions(self) -> list[str]:
        return [self.raw_data]

class MyCompletionFn(CompletionFn):
    def __call__(self, prompt) -> MyCompletionResult:
        return MyCompletionResult("hello")
