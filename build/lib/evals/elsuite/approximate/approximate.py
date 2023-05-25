import random

import evals
import evals.metrics


# Evaluates a model given a prompt and a question, and checks if the model generates the approximately correct NUMBER.
class Approximate(evals.Eval):

    common_prompt = "You are a helpful assistant. If you don't know the answer, give a best guess." \
                    " Answer the following question with a single approximate number and no additional text."
    tolerance = 0.1 # ±10%

    # args provided magically by arithmetic.yaml args:… fields
    def __init__(self, train_jsonl, test_jsonl, train_samples_per_prompt=2, **kwargs):
        super().__init__(**kwargs)
        self.train_jsonl = train_jsonl
        self.test_jsonl = test_jsonl
        self.train_samples_per_prompt = train_samples_per_prompt
        self.common_prompt=kwargs["common_prompt"] or self.common_prompt
        self.common_prompt=kwargs["tolerance"] or self.tolerance

    def run(self, recorder):
        """
        Called by the `oaieval` CLI to run the eval. The `eval_all_samples` method calls `eval_sample`.
        """
        self.train_samples = evals.get_jsonl(self.train_jsonl)
        test_samples = evals.get_jsonl(self.test_jsonl)
        self.eval_all_samples(recorder, test_samples)

        # Record overall metrics
        return {
            "accuracy": evals.metrics.get_accuracy(recorder.get_events("match")),
        }

# If the model fails to understand the instruction "Answer with a single integer number" this will rightfully yield a low score.
    def is_approximate_match(self, expected, sampled):
      ratio: float = int(expected) / int(sampled)
      return ratio > (1-self.tolerance) and ratio < (1+self.tolerance)


    def eval_sample(self, test_sample, rng: random.Random):
        """
        Called by the `eval_all_samples` method to evaluate a single sample.

        ARGS
        ====
        `test_sample`: a line from the JSONL test file
        `rng`: should be used for any randomness that is needed during evaluation

        This method does the following:
        1. Generate a prompt that contains the task statement, a few examples, and the test question.
        2. Check if the model generates the correct answer.
        """

        prompt = [{"role": "system", "content": self.common_prompt}]
        prompt += [{"role": "user", "content": test_sample["problem"]}]

        # evals.check_sampled_text(self.model_spec, prompt, expected=sample["answer"])

        result, actual_prompt, metadata = evals.completion_query(
          prompt=prompt,
          temperature=0.0,
          model_spec=evals.model_spec,
        )
        choice = result["choices"][0]

        sampled = choice["text"].strip() if evals.model_spec.strip_completion else choice["text"]

        expected = test_sample["ideal"]
        match = self.is_approximate_match(expected, sampled)
        # todo: does result["match"] work with a float instead of binary?

        result["expected"] = expected
        result["match"] = match
        result["metadata"] = metadata
        evals.record_sampling(**result)
        evals.record_match(match, expected=expected, picked=expected, sampled=sampled)
