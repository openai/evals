# Error Recovery

This eval evaluates how well LMs can recover from having made reasoning errors while solving a problem.

## Usage

Run with:

```
oaieval <solver> error-recovery
```

We suggest using `error_recovery/gpt-4-0613` or `error_recovery/gpt-3.5-turbo-0613` as default choices for `<solver>`

For more examples of running this eval, see `scripts/run_experiments.sh`

## Dataset

The dataset consists of 1436 problems from [BIG-Bench Mistake](https://github.com/WHGTyen/BIG-Bench-Mistake) which have reasoning traces with mistakes in them, and incorrect answers.

## Evaluation Process

For each example, we run the solver on three variations of the problem:

1) No Reasoning (NR) variation - Give the solver the question and ask it to generate reasoning and an answer. Example below:
```
[SYSTEM]
Solve the given problem, writing your reasoning along the way.

[USER]
Solve this problem for x: x = ((3 + 3) * 5) + 2
```


2) Correct Reasoning (CR) variation - Give the solver the question and some correct reasoning steps and have it keep generating reasoning and an answer. Example below:

```
[SYSTEM]
Solve the given problem, writing your reasoning along the way.

[USER]
Solve this problem for x: x = ((3 + 3) * 5) + 2

[ASSISTANT]
3 + 3 = 6
```

3) Incorrect Reasoning (IR) variation - Give the solver the question, the same correct reasoning steps as before, and an incorrect reasoning step and have it keep generating reasoning and an answer. 

```
[SYSTEM]
Solve the given problem, writing your reasoning along the way.

[USER]
Solve this problem for x: x = ((3 + 3) * 5) + 2

[ASSISTANT]
3 + 3 = 6
6 * 5 = 35
```


## Metrics

The below are the key metrics of this eval:

| Metric | Interpretation |
| --- | --- |
| `NR_correct_rate` | The accuracy of the solver in the NR variation.
| `CR_correct_rate` | The accuracy of the solver in the CR variation.
| `IR_correct_rate` | The accuracy of the solver in the IR variation.

## Variants

| Variant | Notes |
| --- | --- |
| Default: `error-recovery.main` | Default setting. Includes all 1436 examples. |
| `error-recovery.medium` | A balanced subset of 250 examples, 50 from each task. |
| `error-recovery.small` | A balanced subset of 10 examples, 2 from each task (primarily for quick tests). |

## Token Usage Estimates

Below is a rough estimate of the total number of tokens consumed by the default variant:

| Solver | Tokens |
| --- | --- |
| error_recovery/gpt-3.5-turbo-0613 | 4,750,781 |
| error_recovery/gpt-4-0613 | 4,386,440 |

## Version History

- v0: Initial version released

## Contribution statement

Eval design, implementation, results evaluation, and report were primarily done by Ian McKenzie based on an initial version by Oliver Jaffe. Work done under the guidance of (alphabetically by last-name) Steven Adler, James Aung, and Chan Jun Shern, who provided research input, report revisions, and project management support.
