# Already Said That

This eval measures how robust models are to distractors when performing
sequential tasks. We construct a toy task where the model needs to determine
whether it has already seen a given word, and inject distractor questions into
the interaction, keeping track of model performance throughout.

## Usage

Run with:

```bash
oaieval <solver> already_said_that
```

We have found that `generation/direct/gpt-4-0125-preview` works well on this
eval. For more examples of tested solvers, see
[`./scripts/run_experiments.sh`](./scripts/run_experiments.sh).

## Dataset

The dataset consists of 500 samples, where each sample contains 100 unique words
randomly sampled from the [WordNet corpus](https://wordnet.princeton.edu/) via
the `nltk` library.

We also rely on four sets of distractor questions, sourced directly from the
datasets of pre-existing evals. Specifically we make use of the datasets of the
following evals from our evals registry:

- [`which-is-heavier`](../../registry/evals/which-is-heavier.yaml)
- [`first-letters`](../../registry/evals/first-letters.yaml)
- [`ambigous-sentences`](../../registry/evals/ambiguous-sentences.yaml)
- [`reverse-sort-words-eng`](../../registry/evals/reverse-sort-words-eng.yaml)

## Evaluation Process

The evaluation process is as follows for a given sample from our dataset:

1. The `TASK_DESCRIPTION` prompt is shown to the solver.
2. For 100 turns, we either show a word to the solver or a distractor question,
   with probability 2/3 and 1/3 respectively.
3. If a word is shown, we prefix it with `MAIN TASK -`, to indicate that we are
   asking the solver to perform the main task of determining whether it has seen
   the word before.
4. When showing a word, we randomly show previously seen words with a
   probability of 1/2 and new words with a probability of 1/2.
5. If we show a distractor question, we directly show the question to the
   solver.
6. The solver should respond with its answer wrapped in the format
   `[answer: <answer>]`.
7. The solver's response is parsed and compared to the correct answer.
8. If the solver's response is incorrect or a violation is raised (answered in
   the incorrect format), in the case of the main task we stop the interaction
   and record the number of turns the solver lasted for. Otherwise we continue
   to the next turn.

## Prompts

We refer readers to [`./prompts.py`](./prompts.py) for the `TASK_DESCRIPTION`
used in the eval.

We refer readers to [`./distractors.py`](./distractors.py) for any cosmetic
changes we make to the distractor questions.

## Metrics

Below are the metrics returned by the eval:

<!-- prettier-ignore-start -->
| **Metric**              	| **Notes**                                                                                                                                                                                                                                                 	|
|-------------------------	|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	|
| `avg_num_turns`           	| The average number of turns shown before the model fails across the samples. Higher is better. Best possible is 100.                                                                                                                                      	|
| `stddev_num_turns`        	| The standard deviation on the above.                                                                                                                                                                                                                      	|
| `median_num_turns`        	| The median number of turns shown before the model fails across the samples. Higher is better. Best possible is 100.                                                                                                                                       	|
| `max_num_turns`           	| The maximum number of turns shown before the model fails across the samples.                                                                                                                                                                              	|
| `min_num_turns`           	| The minimum number of turns shown before the model fails across the samples.                                                                                                                                                                              	|
| `false_positive_rate`     	| How often the model answers “yes” when it should have answered “no” (i.e. a new word is shown, and the model claims to have seen it already).                                                                                                             	|
| `false_negative_rate`     	| How often the model answers “no” when it should have answered “yes” (i.e. a word is shown again, and the model claims to not have seen it).                                                                                                               	|
| `avg_distractor_accuracy` 	| For a given sample interaction, we measure whether each model response to a given distractor question is accurate. We then compute the accuracy on the distractor questions shown over the interaction. We then average this accuracy across all samples. 	|
| `violation_rate`          	| how often the model responds in an invalid format, i.e. not using the `[answer: <answer>]` format.                                                                                                                                                          |
| `avg_num_distractors`     	| The average number of distractors shown before the model fails across the samples. Higher is better. Best possible is around 33.                                                                                                                          	|
| `stddev_num_distractors`  	| The standard deviation on the above.                                                                                                                                                                                                                      	|
| `median_num_distractors`  	| The median number of distractors shown before the model fails across the samples. Higher is better. Best possible is around 33.                                                                                                                           	|
| `max_num_distractors`     	| The maximum number of distractors shown before the model fails across the samples.                                                                                                                                                                        	|
| `min_num_distractors`     	| The minimum number of distractors shown before the model fails across the samples.                                                                                                                                                                        	|
<!-- prettier-ignore-end -->

## Variants

We consider each of the four distractor datasets mentioned in
[Dataset](#dataset) as a variant of the eval.

```bash
oaieval <solver> already_said_that.<distractor>
```

We also have a `distractorless` variant where we only show words to the solver.
We use this as a baseline to determine how robust the solver is to distractors.

```bash
oaieval <solver> already_said_that.distractorless
```

## Custom Solvers

We implement 2 custom solvers for this eval in [./solvers.py](./solvers.py):

1. `RandomBaselineSolver`: A solver that randomly answers `yes` or `no` for any
   input. We view this baseline as equivalent to randomly guessing.
2. `AlreadySaidThatHuman`: A helper solver class that wraps the `HumanCliSolver`
   class such that users do not have to wrap their answer in the
   `[answer: <answer>]` format and can instead just directly type the answer.

## Token Usage Estimates

Below are approximate token usage estimates for a given run (one run = all
samples) of the eval, for each of the distractor variants.

For Direct gpt-4-0125-preview:

| Distractor variant    | Input      | Output  | Total      |
| --------------------- | ---------- | ------- | ---------- |
| which-is-heavier      | 17,960,000 | 80,000  | 18,040,000 |
| ambiguous-sentences   | 27,750,000 | 110,000 | 27,860,000 |
| first-letters         | 19,850,000 | 80,000  | 19,940,000 |
| reverse-sort-words-en | 10,700,000 | 120,000 | 10,820,000 |
| distractorless        | 27,550,000 | 120,000 | 27,680,000 |

For Direct gpt-3.5-turbo-0125:

| Distractor variant    | Input     | Output | Total     |
| --------------------- | --------- | ------ | --------- |
| which-is-heavier      | 1,200,000 | 10,000 | 1,210,000 |
| ambiguous-sentences   | 1,540,000 | 20,000 | 1,550,000 |
| first-letters         | 2,120,000 | 20,000 | 2,140,000 |
| reverse-sort-words-en | 910,000   | 20,000 | 940,000   |
| distractorless        | 1,250,000 | 20,000 | 1,270,000 |

For Direct gpt-4-base:

| Distractor variant    | Input      | Output    | Total      |
| --------------------- | ---------- | --------- | ---------- |
| which-is-heavier      | 16,950,000 | 3,670,000 | 20,620,000 |
| ambiguous-sentences   | 23,100,000 | 4,390,000 | 27,490,000 |
| first-letters         | 25,310,000 | 4,870,000 | 30,180,000 |
| reverse-sort-words-en | 14,380,000 | 2,760,000 | 17,140,000 |
| distractorless        | 24,460,000 | 5,000,000 | 29,460,000 |

For CoT gpt-4-0125-preview:

| Distractor variant    | Input       | Output    | Total       |
| --------------------- | ----------- | --------- | ----------- |
| which-is-heavier      | 263,600,000 | 1,900,000 | 265,500,000 |
| ambiguous-sentences   | 383,500,000 | 2,700,000 | 386,200,000 |
| first-letters         | 251,700,000 | 1,700,000 | 253,400,000 |
| reverse-sort-words-en | 236,700,000 | 2,100,000 | 238,800,000 |
| distractorless        | 395,500,000 | 2,400,000 | 398,000,000 |

For CoT gpt-3.5-turbo-0125:

| Distractor variant    | Input      | Output  | Total      |
| --------------------- | ---------- | ------- | ---------- |
| which-is-heavier      | 10,100,000 | 190,000 | 10,280,000 |
| ambiguous-sentences   | 7,510,000  | 140,000 | 7,650,000  |
| first-letters         | 16,450,000 | 220,000 | 16,670,000 |
| reverse-sort-words-en | 4,690,000  | 150,000 | 4,840,000  |
| distractorless        | 30,230,000 | 310,000 | 30,540,000 |

## Future modifications

- Extending the range of distractors considered, either by incorporating more
  evals or designing new distractor variants.
- Experiment with multiple distractor sources in a single eval run, to see if
  the variety of distractors affects the model's robustness.

## Version History

- v0: Initial version released

## Contribution Statement

Eval design, implementation, and results evaluation were primarily conducted by
Giulio Starace, under the guidance of (alphabetically by last-name) Steven
Adler, Andrei Alexandru, James Aung, and Chan Jun Shern who provided research
input, report revisions, and project management support.
