from pyparsing import Any
import evals
from evals.api import CompletionFn
from evals.eval import Eval
import evals.metrics


class MultiTurnEval(Eval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        *args,
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)

    def generate_strategy(self, question, completion_fn: CompletionFn) -> str:
        prompt = f"Analyze the problem and form a detailed step by step strategy for answering it below. Do not answer the question!\n\nProblem:\n{question}\n"
        result = completion_fn(prompt)
        return result.get_completions()[0]

    def pick_strategy(self, problem: str, strategies: list[str], completion_fn: CompletionFn) -> str:
        prompt = f"Analyze the strategies below, and pick the best one for solving the problem by ending your response with \"PICK=STRATEGY #0\". If none of them are good, pick the one that is least bad.\n\nProblem:\n{problem}\n\nStrategies:\n\n"
        for i, strategy in enumerate(strategies):
            prompt += f"STRATEGY #{i}:\n{strategy}\n\n"
        result = completion_fn(prompt)
        completion = result.get_completions()[0]
        return completion[completion.find("PICK=STRATEGY #")+1:]

    def eval_sample(self, sample: Any, *_):
        problem = sample[0]
        expected_answer = sample[1]

        # generate strategies
        strategies = [
            self.generate_strategy(problem, completion_fn)
            for completion_fn in self.completion_fns
        ]

        print("Strategies:\n\n")
        for i, strategy in enumerate(strategies):
            print(f"STRATEGY #{i}:\n{strategy}\n\n")

        # ask first completion_fn which one is best
        picked = self.pick_strategy(problem, strategies, self.completion_fns[0])
        picked_idx = int(picked[-1])
        print(f"Picked strategy #{picked_idx}\n\n")

        # ask first completion_fn to generate an explanation using this strategy
        picked_strategy = strategies[picked_idx]
        prompt = f"{problem}\n\n{picked_strategy}\n\nUsing the above strategy, please give an explanation for your answer, and then give your answer as a single letter (A/B/C/D)."
        sampled_answer = self.completion_fns[0](prompt).get_completions()[0]
        print(f"Explanation: {sampled_answer}\n\n")

        # record the result
        evals.record_and_check_match(prompt, sampled_answer, expected=expected_answer)

    def run(self, recorder):
        samples = [
            ("If A = (1, 2, 3, 4). Let ~= {(1, 2), (1, 3), (4, 2)}. Then ~ is\n(A) not anti-symmetric\n(B) transitive\n(C) reflexive\n(D) symmetric", "B"),
        ]
        self.eval_all_samples(recorder, samples)
        return {
            "accuracy": evals.metrics.get_accuracy(recorder.get_events("match")),
        }