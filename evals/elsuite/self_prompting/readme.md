# Eval description

How well can LMs write prompts for themselves to perform various tasks?

In the Self-Prompting eval, models (Prompters) write prompts for other models (Taskers) to perform various tasks -- which are other evals from this repository (listed below). Prompters are given an initial human-written prompt for the task, and asked to rewrite it for a given Tasker model. The effectiveness of the Prompters are measured in terms of the accuracy of downstream Taskers on the tasks. We measure this prompting ability for a variety of different downstream models: gpt-3.5-turbo, gpt-4-base, and gpt-4.

The headline metric for a Prompter’s success is the mean accuracy of the predictions of all its Taskers on all tasks.
- For our primary metric `accuracy`, the accuracy score uses an exact match criterion to judge if the tasker response is correct or not (a response is correct if and only if it exactly matches the true label in the dataset).
- As a secondary metric `accuracy_fuzzy`, we also compute results with a fuzzy match criterion, which counts a response as correct if either the model response contains the label or the label contains the response.

Additionally, we also present `accuracy_improvement_wrt_oriprompt` and `accuracy_fuzzy_improvement_wrt_oriprompt` which are the accuracies normalized relative to the score of the original prompt baseline. This is a score between -1 and +1, where -1 means the current score maximally regresses from the baseline (i.e. the current score is 0), 0 means the current score is the same as the baseline, and +1 means the current score achieves max improvement over the baseline. By default, the baseline score is a cached score of the original prompt (`self_prompting/oriprompt/baseline`) on the `self_prompting.full` eval.

# Usage

To run the eval, use the following command:
```bash
oaieval {solver} self_prompting
```
where `{solver}` is the name of the solver you want to evaluate, e.g. `self_prompting/chat_completion/gpt-4-32k`.

# Experiments
As a starting point for deeper exploration, we provide scripts for comparing various solvers and eval variants, as well as for plotting the results. To run these:
```
cd scripts/
bash run_experiments.sh
```

# Dataset

To form the self-prompting dataset, we extract tasks from this `evals` repository, selecting for datasets with
1. A system prompt that can be straightforwardly converted into a generic instruction for all task samples
2. A straightforward input-output format for each task sample.
3. Designed to be evaluated with an exact match criterion.

The full list of 50 evals we use can be found in `scripts/dataset/eval_list.py`.

# Token estimate
Below, we present a rough estimate of the total number of tokens consumed by the eval, including both input and output tokens.

For self-prompting, each eval run queries multiple models. In the following table, we present the number of tokens consumed by Prompter models:

| Model             | Solver type     | Tokens  |
|-------------------|-----------------|---------|
| code-davinci-002  | completion_hhh  | 400 000 |
| gpt-4-base        | completion_hhh  | 360 000 |
| gpt-3.5-turbo-16k | chat_completion | 180 000 |
| gpt-4-32k         | chat_completion | 155 000 |
| gpt-3.5-turbo-16k | cot             | 480 000 |
| gpt-4-32k         | cot             | 420 000 |
| gpt-3.5-turbo-16k | cotexpert       | 495 000 |
| gpt-4-32k         | cotexpert       | 450 000 |

In addition to the Prompter tokens, each run also queries multiple Tasker models. By default, we use gpt-3.5-turbo, gpt-4-base, and gpt-4, consuming an additional 100k-200k tokens per model.

To calculate dollar cost from token counts, please check the latest token pricing [here](https://openai.com/pricing). Note that we count both input and output tokens together, so a lower and upper estimate of the cost of each variant can be predicted.

# Contribution statement
Eval design, implementation, and results evaluation were primarily conducted by Chan Jun Shern under the guidance of (alphabetically by last-name) Steven Adler, James Aung, Rosie Campbell, and Jade Leung, who provided research input and project management support.
