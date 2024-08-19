# Function Deduction

This eval evaluates how well a model can refine a hypothesis according to new evidence and how well it chooses to gather new information.

In Function Deduction:

- There is a secret mathematical function that maps an integer to another integer.
- The evaluated model interacts with the function by picking inputs to run through the function and observing black-box outputs.
- The model’s goal is to correctly predict outputs for a specified set of inputs, which is only possible by working out the underlying logic of the function.

![fd](https://github.com/openai/policy-research-evals/assets/129281094/6c41be74-8237-4bb3-b0fc-13454c20389c)

## Usage

Run with:

```
oaieval <solver> function_deduction
```

We suggest using `function_deduction/cot/gpt-4-32k` or `function_deduction/cot/gpt-3.5-turbo-16k` as default choices for `<solver>`

See `evals/registry/solvers/function_deduction.yaml` for a full list of recommended Solvers.

For more examples of running this eval, see `scripts/run_experiments.sh`

## Dataset

The dataset are 100 int -> int functions of varying difficulty. Outputs of the functions for inputs 0 through 100 (the allowable range) are pre-cached and in the dataset. During the eval run, the eval simply looks up the output for any input the solver is asking about.

## Evaluation Process

Samples from the dataset are evaluated one-by-one.

A model has up to 20 rounds to guess what the function would return for the three test input integers.

In each round the model can either:

- Ask what the function returns for a given integer (other than the test inputs) *or*
- Make a guess for the values that would be returned by the function for the test inputs.

If the model makes a guess and the guess is correct for all 3 test inputs, the model has solved this sample; otherwise, if it guesses wrong, the model is told whether it guessed any subset of the test inputs correctly.

After 20 rounds without a correct guess, the evaluation for that sample ends and the next sample starts. Rounds where the model answered in an incorrect format are still counted towards the 20 round total.

## Task State

| Key  | Value |
| --- | --- |
| `task_description` | I have an important task for you. Instructions below.<br><br>I have a deterministic function `def calc(x: int) -> int:` that accepts an integer in range [0, 100] as an argument and returns an integer.<br><br>I want you to correctly determine the result of this function for each of the following inputs: {inputs}<br><br>At each turn you must select one of the following 2 options:<br>Option (A): Gather useful information. Ask me what would be returned for a given input to inform your hypothesis of what the function is. \<br>This must be an integer in range [0, 100], other than {inputs}.<br>Option (B): State your guess for the values returned by the function for inputs {inputs}. Your guess must include all three numbers, in this order. \<br>If your guess is correct, you win. If not, the task will continue.<br><br>You have {n_rounds} turns to determine the correct answer, else you lose.<br><br>Your goal is to determine the correct answer in as few turns as possible, so choose your actions carefully!<br>Your final score will be:<br>* If you make the correct guess: {n_rounds} - number_of_rounds_played<br>* If you fail to make the correct guess: -{n_rounds}<br>You should try to score as high as possible.<br><br>You should respond either with a single integer in range 0-100 if you decide to take option A or with three integers if you make the final guess (i.e. use option B).<br>Don't say anything more except for either an integer or 3 integers. |
| `current_state`  | CurrentState object that tracks various data from the current dialog. |

## Metrics

The below are the key metrics of this eval:

| Metric | Interpretation |
| --- | --- |
| `adjusted_avg_score` | Combination metric of the below 2 metrics. The average number of rounds for solved samples, or 40 for not-solved samples. (lower is better) |
| `solved_ratio` | The percentage of solved samples (higher is better) |
| `avg_success_rounds` | The average number of rounds for solved samples (lower is better) |

## Variants

| Variant | Notes |
| --- | --- |
| Default: `function_deduction.easy` | Default setting as described above. 1 trial per sample |
| `function_deduction.easy.long` | 10 trials per sample |
| `function_deduction.easy.dev5` | Dev set with only 5 samples |
| `function_deduction.hard` | A hard variant where the model is only told ‘this guess is incorrect’ if its wrong, instead of being told which inputs it got right/wrong. |
| `function_deduction.hard.dev5` | Dev set with only 5 samples |

## Token Usage Estimates

Below is a rough estimate of the total number of tokens consumed by the default variant:

| Solver | Tokens |
| --- | --- |
| function_deduction/gpt-4-base | 3 840 000 |
| gpt-4-32k | 880 000 |
| gpt-3.5-turbo-16k | 1 560 000 |
| function_deduction/cot/gpt-4-32k | 12 400 000 |
| function_deduction/cot/gpt-3.5-turbo-16k | 13 230 000 |

## Version History

- v0: Initial version released

## Contribution statement

Eval design, implementation, and results evaluation were primarily conducted by Jan Betley with contributions from Andrei Alexandru. Report by James Aung. Work done under the guidance of (alphabetically by last-name) Steven Adler, and Chan Jun Shern, who scoped and managed the broader research project, including input on evaluation design, results analysis, and interpretation.
