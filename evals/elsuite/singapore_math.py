import random
import evals
import evals.metrics
import evals.api

class SingaporeMath(evals.Eval):
    def __init__(self, test_jsonl, **kwargs):
        super().__init__(**kwargs)
        self.test_jsonl = test_jsonl

    def run(self, recorder):
        test_samples = evals.get_jsonl(self.test_jsonl)
        self.eval_all_samples(recorder, test_samples)

        # Record overall metrics
        return {
            "accuracy": evals.metrics.get_accuracy(recorder.get_events("match")),
        }

    def eval_sample(self, test_sample, rng: random.Random):

        # Prepare prompt
        prompt = [{"role": "system", "content": "You are an expert math professor. Work out the answer to the math questions below. Always start by assigning variables to represent the unknowns in the problem. End your explanation with 'Therefore, the answer is  <answer>'. The answer is always a number."}]
        prompt += test_sample["input"]

        # Get model's answer to the question, which allows chain of thought reasoning
        result, _,  _ = evals.api.completion_query(self.model_spec, prompt)
        choice = result["choices"][0]
        explanation = choice["text"].strip() if self.model_spec.strip_completion else choice["text"]

        # Extract the answer from the model's explanation
        # Instructions help ensure formatting of the answer is consistent with the correct answer
        prompt += [{"role": "system", "content": explanation}]
        prompt += [{"role": "user", "content": "Based on the above, what is the answer to the original question? Please only output the answer as a number with no further explanation, words or units. Unless otherwise instructed, if the answer is a fraction, output it as a decimal number only if it is a terminating decimal. Your answer format should be <answer> with no other words."}]

        evals.check_sampled_text(self.model_spec, prompt, expected=test_sample["ideal"])