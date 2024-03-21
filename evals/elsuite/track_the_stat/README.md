# Track the Stat

This eval measures how well models can implicitly keep track of task state, by
asking models to compute the rolling median or the rolling mode over a sequence
of integers.

## Usage

Run with:

```bash
oaieval <solver> track_the_stat
```

We have found that `generation/direct/gpt-4-0125-preview` works well on this
eval. For more examples of tested solvers, see
[`./scripts/run_experiments.sh`](./scripts/run_experiments.sh).

## Evaluation Process

The evaluation process is as follows for a given sample from our dataset:

1. The `TASK_DESCRIPTION` prompt is shown to the solver.
2. The sample contains an integer to use as a seed for a random number
   generator.
3. The random number generator generates 300 random integers between 0 and 100,
   with replacement.
4. The integers are shown one by one to the solver.
5. At each turn (i.e., after each integer is shown), the solver needs to respond
   with the current rolling median or the current rolling mode of the integers
   seen so far.
6. The solver's response is parsed and compared to the correct rolling median or
   rolling mode.
7. If the solver's response is incorrect or a violation is raised (answered in
   the incorrect format), the evaluation stops and we measure how many turns the
   solver lasted for. If the solver's response is correct, we move on to the
   next integer.

## Prompts

We refer readers to the [`./prompts/`](./prompts/) folder for the
`TASK_DESCRIPTION` used in the eval.

## Metrics

Below are the metrics returned by the eval:

<!-- prettier-ignore-start -->
| **Metric**        	| **Notes**                                                                                                                                  	|
|-------------------	|--------------------------------------------------------------------------------------------------------------------------------------------	|
| avg_max_length    	| The maximum sequence length the model can handle before failing, averaged across the samples. Higher is better. Best possible is 300.      	|
| stddev_max_length 	| The standard deviation on the above.                                                                                                       	|
| median_max_length 	| The median of the maximum sequence length the model can handle before failing, across the samples. Higher is better. Best possible is 300. 	|
| max_max_length    	| The maximum sequence length the model handled before failing across all samples.                                                           	|
| min_max_length    	| The minimum sequence length the model handled before failing across all samples.                                                           	|
| violation_rate    	| how often the model responds in an invalid format. i.e. not using the `[<task>: <answer>]` format.                                         	|
<!-- prettier-ignore-end -->

## Variants

The eval has two variants: median and mode. In the median variant, the solver
needs to track the rolling median. In the mode variant, the solver needs to
track the rolling mode.

```bash
oaieval <solver> track_the_stat.<median/mode>
```

## Custom Solvers

We implement 3 custom solvers for this eval in [./solvers.py](./solvers.py)

1. `ExplicitStateSolver`: A nested solver that injects an explicit
   representation of the task state after each number is seen. For example, for
   the median task we inject the sorted list of numbers seen so far. For the
   mode task, we inject a dictionary that maps each number seen so far to its
   count. We view this solver as a baseline for the task, providing the
   performance of the models on _explicit_ state tracking, rather than the
   default _implicit_ state tracking.
2. `RandomBaselineSolver`: A solver that randomly chooses a number from the
   numbers seen so far as the rolling median or mode. In case of even length
   lists in the median variant, it chooses two random numbers and returns their
   arithmetic mean. We view this baseline as equivalent to randomly guessing.
3. `TrackTheStatHuman`: A helper solver class that wraps the `HumanCliSolver`
   class such that users do not have to wrap their answer in the
   `[median: <answer>]` or `[mode: <answer>]` format and can instead just
   directly type the number.

## Token Usage Estimates

Below are token usage estimates for a given run (one run = all samples) of the
eval.

For the mode task:

| Model (state tracking)        | Input     | Output    | Total      |
| ----------------------------- | --------- | --------- | ---------- |
| gpt-3.5-turbo-0125 (implicit) | 670,000   | 10,000    | 680,000    |
| gpt-3.5-turbo-0125 (explicit) | 2,710,000 | 30,000    | 2,740,000  |
| gpt-4-base (implicit)         | 9,030,000 | 2,110,000 | 11,150,000 |
| gpt-4-base (explicit)         | 3,720,000 | 960,000   | 4,680,000  |
| gpt-4-0125-preview (implicit) | 3,050,000 | 30,000    | 3,080,000  |
| gpt-4-0125-preview (explicit) | 8,580,000 | 50,000    | 8,630,000  |

For the median task:

| Model (state tracking)        | Input     | Output  | Total     |
| ----------------------------- | --------- | ------- | --------- |
| gpt-3.5-turbo-0125 (implicit) | 430,000   | 10,000  | 440,000   |
| gpt-3.5-turbo-0125 (explicit) | 880,000   | 10,000  | 890,000   |
| gpt-4-base (implicit)         | 2,900,000 | 760,000 | 3,660,000 |
| gpt-4-base (explicit)         | 3,250,000 | 810,000 | 4,060,000 |
| gpt-4-0125-preview (implicit) | 690,000   | 10,000  | 700,000   |
| gpt-4-0125-preview (explicit) | 1,430,000 | 20,000  | 1,450,000 |

## Future modifications

- Identify new variants of the task beyond median or mode, where the explicit
  state is either impossible to represent or not useful for the task. This would
  allow us to more comfortably measure the implicit state tracking, even on CoT
  solvers.
- Identify more realistic and/or complex tasks.
- Introduce distractors.

## Version History

- v0: Initial version released

## Contribution Statement

Eval design, implementation, and results evaluation were primarily conducted by
Giulio Starace, under the guidance of (alphabetically by last-name) Steven
Adler, Andrei Alexandru, James Aung, and Chan Jun Shern who provided research
input, report revisions, and project management support.
