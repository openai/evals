# How to add a custom eval

**Important: Please note that we are currently not accepting Evals with custom code!** While we ask you to not submit such evals at the moment, you can still submit modelgraded evals with custom modelgraded YAML files.

This tutorial will walk you through a simple example of writing and adding a custom eval. The example eval will test the model's ability to do basic arithmetic. We will assume that you have followed the setup instructions in the [README](../README.md) and gone through the other docs for how to run and build evals.

When writing your own evals, the primary files of interest are:
- `evals/api.py`, which provides common interfaces and utilities used by eval creators to sample from models and process the results,
- `evals/record.py`, which defines the recorder classes which log eval results in different ways, such as to a local JSON file or to a remote Snowflake database, and
- `evals/metrics.py`, which defines various common metrics of interest.

These files provide a suite of tools for writing new evals. Once you have gone through this tutorial, you can see a more realistic example of these tools in action with the [machine translation](../evals/elsuite/translate.py) [eval example](../examples/lafand-mt.ipynb), which also implements custom eval logic in lieu of using an existing template.

## Create your datasets

The first step is to create the datasets for your eval. Here, we will create toy train and test sets of just two examples each. The test examples are what we will evaluate the model on, and we'll include the train examples as few-shot examples in the prompt to the model.

We will use the new chat format described [here](https://platform.openai.com/docs/guides/chat/introduction). By default, we encourage all evals to be written using chat formatting if you want to evaluate our new models. Under the hood, we [convert](../evals/prompt/base.py) chat formatted data into raw strings for older non chat models.

To create the toy datasets, in your terminal, type:
```bash
echo -e '{"problem": "2+2=", "answer": "4"}\n{"problem": "4*4=", "answer": "16"}' > /tmp/train.jsonl
echo -e '{"problem": "48+2=", "answer": "50"}\n{"problem": "5*20=", "answer": "100"}' > /tmp/test.jsonl
```

## Create an eval

The next step is to write a Python class that represents the actual evaluation. This class uses your datasets to create prompts, which are passed to the model to generate completions. Evaluation classes generally will inherit from the `evals.Eval` base class (defined in `evals/eval.py`) and will override two methods: `eval_sample` and `run`.

Let's create a file called `arithmetic.py` under the `evals/elsuite` folder. We'll start by defining the eval class. Its `__init__` method will take in the arguments we need (references to the train and test sets) along with other `kwargs` that will be handled by the base class. We'll also define the `run` method which takes in a `recorder` and returns the final metrics of interest.

```python
import random
import textwrap

import evals
import evals.metrics

class Arithmetic(evals.Eval):
    def __init__(self, train_jsonl, test_jsonl, train_samples_per_prompt=2, **kwargs):
        super().__init__(**kwargs)
        self.train_jsonl = train_jsonl
        self.test_jsonl = test_jsonl
        self.train_samples_per_prompt = train_samples_per_prompt

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
```

Generally, most `run` methods will follow the same pattern shown here: loading the data, calling `eval_all_samples`, and aggregating the results (in this case, using the `get_accuracy` function in `evals/metrics.py`). `eval_all_samples` takes in both the `recorder` and the `test_samples` and, under the hood, will call the `eval_sample` method on each sample in `test_samples`. So let's write that `eval_sample` method now:

```python
    def eval_sample(self, test_sample, rng: random.Random):
        """
        Called by the `eval_all_samples` method to evaluate a single sample.

        ARGS
        ====
        `test_sample`: a line from the JSONL test file
        `rng`: should be used for any randomness that is needed during evaluation

        This method does the following:
        1. Generate a prompt that contains the task statement, a few examples, and the test question.
        2. Generate a completion from the model.
        2. Check if the generated answer is correct.
        """
        stuffing = rng.sample(self.train_samples, self.train_samples_per_prompt)

        prompt = [
            {"role": "system", "content": "Solve the following math problems"},
        ]

        for i, sample in enumerate(stuffing + [test_sample]):
            if i < len(stuffing):
                prompt += [
                    {"role": "system", "content": sample["problem"], "name": "example_user"},
                    {"role": "system", "content": sample["answer"], "name": "example_assistant"},
                ]
            else:
                prompt += [{"role": "user", "content": sample["problem"]}]


        result = self.completion_fn(prompt=prompt, temperature=0.0, max_tokens=1)
        sampled = result.get_completions()[0]

        evals.record_and_check_match(prompt=prompt, sampled=sampled, expected=sample["answer"])
```
You'll notice that `eval_sample` doesn't take the `recorder` as an argument. This is because `eval_all_samples` sets it to be the default recorder before calling `eval_sample`, and the recording utilities defined in `evals/record.py` use the default recorder. In this example, the `eval_sample` method passes off a lot of the heavy lifting to the `evals.check_sampled_text` utility function, which is defined in `evals/api.py`. This utility function queries the model, defined by `self.model_spec`, with the given `prompt` and checks to see if the result matches the `expected` answer (or one of them, if given a list). It then records these matches (or non matches) using the default recorder.

`eval_sample` methods may vary greatly based on your use case. If you are building custom evals, it is a good idea to be familiar with the functions available to you in `evals/record.py`, `evals/metrics.py`, and especially `evals/api.py`.

## Register your eval

The next step is to register your eval in the registry so that it can be run using the `oaieval` CLI.

Let's create a file called `arithmetic.yaml` under the `evals/registry/evals` folder and add an entry for our eval as follows:

```yaml
# Define a base eval
arithmetic:
  # id specifies the eval that this eval is an alias for
  # in this case, arithmetic is an alias for arithmetic.dev.match-v1
  # When you run `oaieval davinci arithmetic`, you are actually running `oaieval davinci arithmetic.dev.match-v1`
  id: arithmetic.dev.match-v1
  # The metrics that this eval records
  # The first metric will be considered to be the primary metric
  metrics: [accuracy]
  description: Evaluate arithmetic ability
# Define the eval
arithmetic.dev.match-v1:
  # Specify the class name as a dotted path to the module and class
  class: evals.elsuite.arithmetic:Arithmetic
  # Specify the arguments as a dictionary of JSONL URIs
  # These arguments can be anything that you want to pass to the class constructor
  args:
    train_jsonl: /tmp/train.jsonl
    test_jsonl: /tmp/test.jsonl
```

The `args` field should match the arguments that your eval class `__init__` method expects.

## Run your eval

The final step is to run your eval and view the results.

```sh
pip install .  # you can omit this if you used `pip install -e .` to install
oaieval gpt-3.5-turbo arithmetic
```

If you run with the `gpt-3.5-turbo` model, you should see an output similar to this (we have cleaned up the output here slightly for readability):

```
% oaieval gpt-3.5-turbo arithmetic
... [registry.py:147] Loading registry from .../evals/registry/evals
... [registry.py:147] Loading registry from .../.evals/evals
... [oaieval.py:139] Run started: <run_id>
... [eval.py:32] Evaluating 2 samples
... [eval.py:138] Running in threaded mode with 1 threads!
100%|██████████████████████████████████████████| 2/2 [00:00<00:00,  3.35it/s]
... [record.py:320] Final report: {'accuracy': 1.0}. Logged to /tmp/evallogs/<run_id>_gpt-3.5-turbo_arithmetic.jsonl
... [oaieval.py:170] Final report:
... [oaieval.py:172] accuracy: 1.0
... [record.py:309] Logged 6 rows of events to /tmp/evallogs/<run_id>_gpt-3.5-turbo_arithmetic.jsonl: insert_time=2.038ms
```
