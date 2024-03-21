# Skill acquisition

This eval tests models' ability to learn a skill with minimal human involvement. In the initial release, models are evaluated on questions related to the [Miskito language](https://en.wikipedia.org/wiki/Miskito_language). Some samples are translation and others are language manipulation exercises. 

## Usage
Run with:
```bash
oaieval <solver> skill_acquisition.miskito
```

Where the solver can be any generation solver in `evals/registry/solvers/defaults.yaml`, eg. `generation/cot/gpt-3.5-turbo-16k`.

## Evaluation process
Every time the eval is run, the model is evaluated twice. The first time, it answers the question directly using whatever prompting technique is executed by the solver you choose. The second time the model runs in a loop, interacting with an interface which gives it access to a knowledge base. The knowledge base contains text files, some of which are relevant for answering the question, while others are unrelated. If models can use this interface to increase their performance on the task, we can say that they've improved or acquired their language translation and manipulation skills.

## Prompts
See `skill_acquisition/utils.py` to review/adjust the prompts used in this eval.

## Datasets

The dataset is generated from [this language course](https://en.wikibooks.org/wiki/Miskito), which comprises 229 questions. We further split this into manipulation-only (`miskito_test_manipulation.jsonl`) and translation-only (`miskito_test_translation.jsonl`) subsets.

## Variants

We test zero-shot and few-shot prompting techniques on the dataset:

| Dataset | Zero-shot | Few-shot |
| --------- | -------- | -------- |
| Miskito | `skill_acquisition.miskito.zero-shot.full`|`skill_acquisition.miskito.few-shot.full`|

The `full` in this case refers to the size of the dataset – there are also variants for testing where only 5 examples are considered, called `dev5`. For full details, look at `evals/registry/skill_acquisition/skill_acquisition.yaml`.

For the few-shot setting, use the eval-specific solvers in `evals/registry/solvers/skill_acquisition.yaml` to avoid train/test leakage.

## Token Usage Estimates

Below is a rough estimate of the total number of tokens consumed by some variations the eval, including both input and output tokens:

| Model | Solver | Prompt tokens | Completion tokens | Total tokens
| --- | --- | --- | --- | --- |
| gpt-3.5-turbo | direct | 1,000,000 | 23,000 | 1,050,000 |
| gpt-3.5-turbo | cot | 930,000 | 120,000 | 1,050,000 |
| gpt-3.5-turbo | fewshot | 450,000 | 9,600 | 460,000 |
| gpt-3.5-turbo-16k | direct | 1,400,000 | 24,000 | 1,500,000 |
| gpt-3.5-turbo-16k | cot | 2,000,000 | 120,000 | 2,100,000 |
| gpt-3.5-turbo-16k | fewshot | 610,000 | 10,000 | 620,000 |
| gpt-4-base | direct | 1,800,000 | 420,000 | 2,200,000 |
| gpt-4-base | cot | 4,700,000 | 890,000 | 5,600,000 |
| gpt-4-base | fewshot | 1,400,000 | 320,000 | 1,700,000 |
| gpt-4-1106-preview | direct | 1,700,000 | 100,000 | 1,800,000 |
| gpt-4-1106-preview | cot | 1,600,000 | 99,000 | 1,700,000 |
| gpt-4-1106-preview | fewshot | 1,700,000 | 95,000 | 1,800,000 |
| gpt-4-32k | direct | 1,800,000 | 80,000 | 1,900,000 |
| gpt-4-32k | cot | 2,700,000 | 180,000 | 2,900,000 |
| gpt-4-32k | fewshot | 190,000 | 6,000 | 190,000 |

## Version History
v0: Initial version released


## Contribution statement

Eval design, implementation, and results evaluation were primarily conducted by Andrei Alexandru. Giulio Starace was responsible for code reviews throughout the implementation process, along with fine-grained feedback on the project in general. Additional guidance was provided by (alphabetically by last-name) Steven Adler, James Aung and Chan Jun Shern, who scoped and managed the broader research project, including input on evaluation design, results analysis, and interpretation.

