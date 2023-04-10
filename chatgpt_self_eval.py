import os
import json
import numpy as np
import evals
import openai
import difflib
import jsonlines
from evals.elsuite import utils
from evals.record import RecorderBase


class CoherentConversation(evals.Eval):
    def __init__(
        self,
        model_specs: evals.ModelSpecs,
        samples_jsonl: str,
        *args,
        max_tokens: int = 500,
        coherence_threshold: float = 0.5,
        **kwargs,
    ):
        super().__init__(model_specs, *args, **kwargs)
        self.max_tokens = max_tokens
        self.samples_jsonl = samples_jsonl
        self.coherence_threshold = coherence_threshold
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def eval_sample(self, test_sample, rng):
        conversation = [test_sample["input"]]
        for i in range(test_sample["num_turns"]):
            generated_response = openai.Completion.create(
                engine=self.model_spec.model,
                prompt=conversation[-1],
                max_tokens=self.max_tokens,
                n=1,
                stop=None,
                temperature=0.5,
            )[0].text.strip()
            conversation.append(generated_response)
            coherence_score = utils.coherence_score(conversation)
            if coherence_score < self.coherence_threshold:
                evals.record.record_metrics(
                    coherence_score=coherence_score,
                    success=False,
                    num_turns=i + 1,
                )
                return False

        evals.record.record_metrics(
            coherence_score=coherence_score,
            success=True,
            num_turns=test_sample["num_turns"],
        )
        return True

    def run(self, recorder: RecorderBase):
        # Load samples
        samples = evals.get_jsonl(self.samples_jsonl)

        # Evaluate each sample
        self.eval_all_samples(recorder, samples)

        # Compute overall metrics and return as results
        num_successes = len([e for e in recorder.get_metrics("success") if e])
        num_failures = len([e for e in recorder.get_metrics("success") if not e])
        success_rate = num_successes / (num_successes + num_failures)
        average_num_turns = np.mean(recorder.get_metrics("num_turns"))
        average_coherence_score = np.mean(recorder.get_metrics("coherence_score"))
        return {
            "success_rate": success_rate,
            "average_num_turns": average_num_turns,
            "average_coherence_score": average_coherence_score,
        }

class MyModelSpecs:
    def __init__(self, model_id="davinci", max_tokens=1024, temperature=0.5):
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model_name = "gpt3"
        self.model_size = "large"
        self.max_length = max_tokens
        self.top_p = 0.9
        self.frequency_penalty = 0
        self.presence_penalty = 0
        self.stop_sequence = None
        self.model = None
        self.name = f"{self.model_name}_{self.model_size}_max_tokens_{self.max_tokens}_temp_{self.temperature}"



class MyEval:
    def __init__(self, model_specs: MyModelSpecs, samples_jsonl: str, *args, **kwargs):
        self.model_spec = model_specs
        self.samples = []
        with jsonlines.open(samples_jsonl) as reader:
            for obj in reader:
                self.samples.append(obj)

    def score_response(self, response, expected):
        raise NotImplementedError

    def evaluate(self):
        scores = []
        for sample in self.samples:
            input_str = sample["input"]
            expected_str = sample["output"]
            response = self.generate_response(input_str)
            score = self.score_response(response, expected_str)
            scores.append(score)
        return sum(scores) / len(scores)

    def generate_response(self, input_str):
        response = openai.Completion.create(
            engine=self.model_spec.model_id,
            prompt=input_str,
            max_tokens=self.model_spec.max_tokens,
            temperature=self.model_spec.temperature,
        )
        return response.choices[0].text.strip()

    def load_samples(self):
        """
        Loads the samples from the JSONL file into a list of tuples containing (input, expected_output).
        """
        samples = []
        with open(self.samples_jsonl, "r") as f:
            for line in f:
                sample = json.loads(line)
                input_text = sample["input"]
                expected_output = sample["output"]
                samples.append((input_text, expected_output))
        return samples



class MyMatch(MyEval):
    def __init__(self, model_specs: MyModelSpecs, samples_jsonl: str, *args, **kwargs):
        super().__init__(model_specs, samples_jsonl, *args, **kwargs)
        self.samples_jsonl = samples_jsonl
    
    def run(self):
        return self.evaluate()

    def score_response(self, response, expected):
        if response.lower() == expected.lower():
            return 1.0
        else:
            return 0.0


class MyIncludes(MyEval):
    def __init__(self, model_specs: MyModelSpecs, samples_jsonl: str, *args, **kwargs):
        super().__init__(model_specs, samples_jsonl, *args, **kwargs)
        self.samples_jsonl = samples_jsonl
    
    def load_samples(self):
        with open(self.samples_jsonl, "r") as f:
            samples = [json.loads(line) for line in f]
        return samples
    
    def score_response(self, response, expected):
        if expected.lower() in response.lower():
            return 1.0
        else:
            return 0.0

    def run(self):
        """
        Runs the evaluation on the samples and returns the average score.
        """
        # Load the samples from the JSONL file
        samples = self.load_samples()

        # Generate responses and compute scores for each sample
        scores = []
        for input_text, expected_output in samples:
            response = self.generate_response(input_text)
            score = self.score_response(response, expected_output)
            scores.append(score)

        # Compute the average score and return it
        avg_score = sum(scores) / len(scores)
        return avg_score



class MyFuzzyMatch(MyEval):
    def __init__(self, model_specs: MyModelSpecs, samples_jsonl: str, *args, **kwargs):
        super().__init__(model_specs, samples_jsonl, *args, **kwargs)
        self.samples_jsonl = samples_jsonl

    def score_response(self, response, expected):
        """
        Scoring function for evaluating generated response against expected output.

        Returns a score between 0 and 1, where higher scores indicate better matches.
        """
        # Use OpenAI's fuzzy search API to compare response and expected output
        response_results = openai.Completion.create(
            engine="davinci",
            prompt=response,
            max_tokens=32,
            temperature=0.5,
            n = 1,
            stop=None,
        )

        expected_results = openai.Completion.create(
            engine="davinci",
            prompt=expected,
            max_tokens=32,
            temperature=0,
            n = 1,
            stop=None,
        )

        # Get a ratio of how similar the response and expected strings are
        # using the Difflib library's SequenceMatcher.
        similarity_ratio = difflib.SequenceMatcher(None, response.lower(), expected.lower()).ratio()

        # A similarity ratio of 1.0 means the strings are identical,
        # while 0.0 means there is no similarity.
        return similarity_ratio

    def run(self):
        """
        Runs the evaluation on the samples in the input JSONL file.

        Returns a dictionary with the results of the evaluation.
        """
        # Load samples from the input JSONL file
        samples = self.load_samples()

        # Initialize variables to track overall score and number of samples
        total_score = 0
        num_samples = len(samples)

        # Evaluate each sample and compute the overall score
        for sample in samples:
            prompt = sample["prompt"]
            expected = sample["completion"]
            response = self.generate_response(prompt)
            score = self.score_response(response, expected)
            total_score += score

        # Compute the final score as the average of the sample scores
        final_score = total_score / num_samples

        # Create a dictionary with the evaluation results
        results = {
            "model": self.model_spec.name,
            "metric": "fuzzy match",
            "score": final_score
        }

        return results

    def load_samples(self):
        samples = []
        with open(self.samples_jsonl, "r") as f:
            for line in f:
                sample = json.loads(line.strip())
                samples.append(sample)
        return samples





class MyEvalRunner:
    """
    A class for running multiple evaluations and collecting the results.
    """

    def __init__(self):
        self.evaluations = []

    def add_evaluation(self, evaluation):
        """
        Add an evaluation to the runner.

        Args:
            evaluation (MyEval): An instance of a MyEval subclass.
        """
        self.evaluations.append(evaluation)

    def run(self):
        """
        Run all the evaluations and return the results.

        Returns:
            dict: A dictionary containing the results of all the evaluations.
        """
        results = {}
        for evaluation in self.evaluations:
            results[evaluation.__class__.__name__] = evaluation.run()
        return results


def accuracy(preds, labels):
    """
    Computes the accuracy of the predictions given the true labels.

    Args:
        preds (list[int]): A list of predicted labels.
        labels (list[int]): A list of true labels.

    Returns:
        float: The accuracy of the predictions.
    """
    assert len(preds) == len(labels)
    num_correct = sum(1 for pred, label in zip(preds, labels) if pred == label)
    return num_correct / len(preds)


# set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# define the model specs
model_specs = MyModelSpecs("davinci")


# Get the directory of the script being executed
script_dir = os.path.dirname(os.path.abspath(__file__))

# Specify the relative path to the JSONL files
samples_dir = "samples"

# Below some lines are commented out to avoid exceeding max_tokens.

# Define the paths to the sample datasets
#match_samples_jsonl = os.path.join(script_dir, samples_dir, "loads.jsonl")
includes_samples_jsonl = os.path.join(script_dir, samples_dir, "load_and_capacity_comparison_evaluation.jsonl")
fuzzy_match_samples_jsonl = os.path.join(script_dir, samples_dir, "influence_line_comparison_evaluation.jsonl")

# create instances of each evaluation class
#match_eval = MyMatch(model_specs, match_samples_jsonl)
includes_eval = MyIncludes(model_specs, includes_samples_jsonl)
fuzzy_match_eval = MyFuzzyMatch(model_specs, fuzzy_match_samples_jsonl)

# create an EvalRunner instance and add each evaluation
runner = MyEvalRunner()
#runner.add_evaluation(match_eval)
runner.add_evaluation(includes_eval)
runner.add_evaluation(fuzzy_match_eval)

# run the evaluations and get the results
results = runner.run()
