# Identifying Variables

This eval tests how well models can determine what should be treated as the
independent, dependent, and control variables for an experiment that tests a
particular hypothesis, given some observational context.

## Usage

Run with:

```bash
oaieval <solver> identifying_variables
```

We have found that `generation/cot/gpt-4-1106-preview` works well on this eval. For more examples of tested solvers, see [`./scripts/run_experiments.sh`](./scripts/run_experiments.sh).

## Evaluation Process

The evaluation process is as follows for a given sample from our dataset:

1. The `TASK_DESCRIPTION` prompt is shown to the solver.
2. The sample is passed through a _renderer_ that processes the samples and
   renders an observation of the interactions of variables, which is placed in
   the `SAMPLE_MESSAGE` prompt template.
3. The solver answers in the form: `[@ANSWER valid_hyp: <true/false>; independent: <var>; dependent: <var>; control: <vars>]`. The answer is parsed and evaluated by the eval. If the answer cannot be parsed, we mark this as a violation and the sample is treated as incorrect.

## Prompts

We refer readers to the [`./prompts.py`](./prompts.py) file for the
`TASK_DESCRIPTION` and `SAMPLE_MESSAGE` prompts used in the eval.

## Metrics

<!-- prettier-ignore-start -->
| **Metric** | **Notes** |
|---|---|
| `ctrl_nDCG` | A modified version of the [normalized discounted cumulative gains (nDCG)](https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG) metric, which rewards listing the  correct control variables first and penalizes naming irrelevant variables. |
| `ctrl_recall` | Number of variables correctly marked as control variables / total number of variables to control according to the gold label |
| `ctrl_recall` | Number of variables incorrectly marked as control variables / total number of variables not to control according to the gold label |
| `hyp_valid_acc` | Target hypothesis plausibility validation accuracy (correct/incorrect) |
| `ind_acc` | Independent variable determination accuracy (correct/incorrect) |
| `dep_acc` | Dependent variable determination accuracy (correct/incorrect) |
| `violation_rate` | Number of samples with violations (model failed to answer in correct format) / total number of samples |
<!-- prettier-ignore-end -->

## Variants

We support variations on the eval along two dimensions, `renderer` and `dataset`:

```bash
oaieval <solver> identifying_variables.<renderer>.<dataset>
```

The eval defaults to `identifying_variables.language-corrset.balanced-ctrl`.

### Dataset

We provide 4 dataset variants:

| `dataset` | Notes |
| --- | --- |
| `balanced-ctrl` | 500 samples balanced across number of control variables (from 0 to 8). |
| `balanced-ctrl-large` | As `balanced-ctrl`, but with 5,000 samples. |
| `balanced-hypotheses` | 500 samples balanced across target hypotheses being implausible/plausible. |
| `balanced-hypotheses-large` | As `balanced-hypotheses`, but with 5,000 samples. |

### Renderers

We have 6 different renderers, implemented in [`./renderers/`](./renderers/).

The default renderer is `language-corrset`. Here is an example render from this type:
```
The following is a description of the observations made about a set of variables.

In general, there were cases where some variables changed in tandem with each other, while others did not.
For example, changes in x_5075 were observed to reflect changes in x_3314 and viceversa.
Changes in x_9549 were not observed to reflect any changes in previously mentioned variables.
Changes in x_1808 were not observed to reflect any changes in previously mentioned variables.
Likewise, changes in x_9726 were observed to reflect changes in x_1808 and viceversa.
```

### Show Tree

We provide an additional variant of the eval where the decision tree implementing
the reasoning for scoring a perfect score is shown to the model. This variant
can be run by passing the `show_tree=True` flag to eval, e.g.

```bash
oaieval <solver> identifying_variables --extra_eval_params show_tree=True
```

## Custom Solvers

We implement two custom programmatic solvers to serve as baselines.

1. `identifying_variables/random`: a solver that randomly selects whether the
   hypothesis is plausible with probability 0.5, and if so randomly samples the
   independent, dependent and control variables. We view this baseline as
   equivalent to randomly guessing.
2. `identifying_variables/noctrl`: this is a solver that always outputs an empty
   list for the variables to control, essentially eliminating any chance of
   false positives. This can provide stronger performance than the random
   baseline, since it avoids any penalization for returning incorrect variables,
   and can even achieve a perfect score on samples that indeed do not have any
   variables to control

We refer to [`./solvers.py`](./solvers.py) for the implementation of these
solvers.

## Token Usage Estimates

We estimated per-run token usage on the default dataset size (500 samples)
for the least and most token-intensive configurations for each model type
(respectively, direct models on `identifying_variables.corrset` with
`show_tree=False`; and CoT models on `identifying_variables.language-tabular`
with `show_tree=True`).

<!-- prettier-ignore-start -->
|  | **input tokens/run** | **output tokens/run** | **total tokens/run** |
|---|---|---|---|
| **GPT-4-base HHH (corrset, no tree)** | 1,200,000 | 250,000 | 1,450,000 |
| **GPT-4-base CoT HHH (language-tabular, with tree)** | 1,500,000 | 240,000 | 1,740,000 |
| **GPT-3.5-turbo Direct (corrset, no tree)** | 430,000 | 88,000 | 518,000 |
| **GPT-3.5-turbo CoT (language-tabular, with tree)** | 780,000 | 14,000 | 794,000 |
| **GPT-4-1106-preview Direct (corrset, no tree)** | 430,000 | 53,000 | 483,000 |
| **GPT-4-1106-preview CoT (language-tabular, with tree)** | 860,000 | 14,000 | 874,000 |
<!-- prettier-ignore-end -->

These estimates were taken using the `balanced-hypotheses` dataset but should
roughly apply to the `-balanced-ctrl` datasets. For `-large` datasets (5000
samples), multiply the above numbers by 10.

## Future modifications

- Revisit the definition of the headline `ctrl_nDCG` metric
- Devise additional auxiliary metrics to paint a more complete picture
- What if we show the decision trees described in natural language rather than
  pseudocode?
- How can we extend this eval to multi-variable dependencies?

## Version History

- v0: Initial version released

## Contribution Statement

Eval design, implementation, and results evaluation and writeup were primarily
conducted by Giulio Starace. James Aung was of enormous assistance in report
writing, and additionally provided general feedback and project management
throughout the eval. Oliver Jaffe and Jan Betley were responsible for code
reviews throughout the implementation process, along with fine-grained feedback
on the project in general. Additional guidance was provided by (alphabetically
by last-name) Steven Adler and Chan Jun Shern, who helped with brainstorming,
gave research input and report revisions.

## Appendix

### Perfect output decision trees

The following decision trees are used to determine the perfect output (aka "gold
label") for a given sample.

---

<img src="images/control_var_tree.png" width="700">

**Figure A1**: Decision tree for determining whether a given variable should be
controlled.

---

<img src="images/valid_hyp_tree.png" width="312">

**Figure A2**: Decision tree for determining a hypothesis is valid and if so
what the independent and dependent variables are.

---
