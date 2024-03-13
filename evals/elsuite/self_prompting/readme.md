# Self Prompting

How well can LMs write prompts for themselves to perform various tasks?

In the Self-Prompting eval, models (Prompters) write prompts for other models (Taskers) to perform various tasks – which are other evals from this repository (listed below). Prompters are given an initial human-written prompt for the task, and asked to rewrite it for a given Tasker model. 

The effectiveness of the Prompters are measured in terms of the accuracy of downstream Taskers on the tasks. This eval measures this prompting ability for a variety of different downstream models: gpt-3.5-turbo, gpt-4-base, and gpt-4.

## Usage

Run with:

```
oaieval <solver> self_prompting
```

We suggest using `self_prompting/direct/gpt-4-32k` or `self_prompting/direct/gpt-3.5-turbo-16k` as default choices for `<solver>`.

See evals/registry/solvers/self_prompting.yaml for a full list of recommended Solvers.

For more examples of running this eval, see `scripts/run_experiments.sh`

## Evaluation Process

In the Self-prompting Eval, models (***Prompters***) write prompts for other models (***Taskers***) to perform various tasks.

For a given task, the Prompter is given key information including a basic task instruction, examples of inputs and corresponding labels, which model it is prompting for, and how its prompt will be presented to that model.

Given this setup, the Prompter generates a model-written prompt.

The generated prompt is directly evaluated by running the Tasker model on the task with that prompt: For each sample in the test dataset, the model-written prompt is concatenated with the input sample and fed to the Tasker model. The output of the Tasker model is judged against the ground-truth label of that sample.

## Task State

The following task state is given to Prompter models:
| Key  | Value |
| --- | --- |
| task_description | Please design a prompt for a large language model to excel on a given evaluation task. Your goal is to help the model achieve as high a score as possible on the evaluation task.<br><br>THE TASK<br>--------<br>Here are some basic instructions that have been written for the task:<br><br>{instruction}<br><br><br>The basic instructions provided above may be incomplete or contain errors. For clarity, we provide some examples of inputs and the output label for the task below. When in doubt, pay attention to these examples and adjust your prompt such that the target model gives its output as demonstrated:<br><br>{samples}<br><br><br>Evaluation criteria: The target model will be tested on new samples that are not shown above, but follow the same task rules. The correctness of the model's output per sample is determined via exact match with the sample's output label. The final score is the accuracy of the target model on all samples (i.e. the number of samples for which the model's output exactly matches the output label, divided by the number of samples).<br><br>PROMPTING THE MODEL<br>-------------------<br>The target model you are designing a prompt for is {tasker_model}.<br><br>Each task sample will be fed independently to the model with your prompt wrapping it. Specifically, your prompt MUST contain at least one instance of the string "[sample_in]" (including brackets, no quotes). This string will be replaced by an input sample from the task before it is passed to the downstream model.<br><br>Your prompt can contain any information you want (e.g. instructions, strategies, formatting tips).<br><br>YOUR RESPONSE<br>-------------<br>Please respond with the prompt for the model. Any text you return here will be filled with the sample input and fed to the model. |
| current_state | {<br>"instruction": # instruction for the task<br>"samples": # training samples for the task<br>"tasker_model": # model that performs the task<br>} |
- {instruction} is the original task description from the dataset.
- {samples_preview} is a JSON-formatted dump of 5 samples from the training dataset (each sample contains the input “question” and output label).
- {tasker_model} is the ID of the Tasker model, e.g. `gpt-3.5-turbo`.

## Dataset

The dataset are 50 tasks drawn from this `evals` repository.

Tasks are selected for
- A system prompt that can be straightforwardly converted into a generic instruction for all task samples
- A straightforward input-output format for each task sample
- Designed to be evaluated with an exact match criterion

The dataset of 50 tasks contains:

1. A generic instruction for that task
2. 10 test samples (each sample is a question and answer pair)
3. Remaining samples are kept as a training set (e.g. for methods to do few-shot learning)

The full list of 50 evals used can be found in `scripts/dataset/eval_list.py`.

## Metrics

| Metric | Interpretation |
| --- | --- |
| `accuracy`  | The mean accuracy of the predictions of all its Taskers on all tasks. Exact match criterion to judge if the tasker response is correct or not (a response is correct if and only if it exactly matches the true label in the dataset |
| `accuracy_fuzzy` | Same as above, but with a fuzzy match criterion, which counts a response as correct if either the model response contains the label or the label contains the response. |
| `prompt_rule_violation_rate` | % samples with rule vioations |
| `n_samples` | Number of samples |
| `accuracy_improvement_wrt_oriprompt` | Accuracy normalised relative to score of the original prompt baseline. This is a score between -1 and +1, where -1 means the current score maximally regresses from the baseline (i.e. the current score is 0), 0 means the current score is the same as the baseline, and +1 means the current score achieves max improvement over the baseline |
| `accuracy_fuzzy_improvement_wrt_oriprompt` | Same as above, but with fuzzy_match |

## Variants

| Variant | Notes |
| --- | --- |
| Default: `self_prompting.full` | 50 tasks with 10 samples per task |
| `self_prompting.small` | 50 tasks with 1 sample per task, only using gpt-3.5-turbo as Tasker |

## Token Usage Estimates

For self-prompting, each eval run queries multiple models.

Number of tokens consumed by Prompter models:

| Model | Solver type | Tokens |
| --- | --- | --- |
| code-davinci-002 | completion_hhh | 400 000 |
| gpt-4-base | completion_hhh | 360 000 |
| gpt-3.5-turbo-16k | chat_completion | 180 000 |
| gpt-4-32k | chat_completion | 155 000 |
| gpt-3.5-turbo-16k | cot | 480 000 |
| gpt-4-32k | cot | 420 000 |
| gpt-3.5-turbo-16k | cotexpert | 495 000 |
| gpt-4-32k | cotexpert | 450 000 |

In addition to the Prompter tokens, each run also queries multiple Tasker models. By default, this eval uses gpt-3.5-turbo, gpt-4-base, and gpt-4, consuming an additional 100k-200k tokens per model.

## Version History

- v0: Initial version released

## Contribution statement

Eval design, implementation, and results evaluation were primarily conducted by Chan Jun Shern under the guidance of (alphabetically by last-name) Steven Adler, James Aung, Rosie Campbell, and Jade Leung, who provided research input and project management support.
