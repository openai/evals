# 20 Questions

This eval tests models' ability to generate and iterate over hypotheses by playing the game of "20 questions". In 20 questions, one of the players – the "gamemaster" – thinks of a word (in our case a noun) and the other player needs to guess it. To help them guess, the player can ask up to 20 yes-or-no questions, which the gamemaster must answer.

## Usage
Run with:
```bash
# Standard variant.
oaieval <solver> twenty_questions.full

# Shortlist variant.
oaieval <solver> twenty_questions.shortlist.full
```

Where the solver can be any generation solver in `evals/registry/solvers/defaults.yaml`, e.g. `generation/cot/gpt-3.5-turbo-16k`, or the chain-of-thought solvers in `evals/registry/solvers/twenty_questions.yaml`.

## Evaluation process
We run a dialogue loop between two models for each sample: the evaluated model and the "gamemaster". By default, the gamemaster is gpt-4-turbo-preview – but this can be updated by specifying a different solver in `evals/registry/evals/twenty_questions.yaml`. 

The dialogue continues until the word is guessed correctly, or until 20 questions have been asked, whichever comes first. We also terminate conversations that last longer than 40 replies, to ensure that models which do not ask questions don't have infinite conversations. Both the maximum questions and the maximum replies can be controlled from the eval YAML file.

## Task State
The task state can be found in `twenty_questions/utils.py`; it reads:
```
You are about to play the game '20 questions'. The other player has thought of a noun which you should try to guess. You can ask 20 yes/no questions, to which they will answer 'yes', 'no', or 'skip' (if they cannot answer your question). You should try to guess the word as soon as possible, using the least amount of questions. To guess a word, type [GUESS <word>] – for example to guess the word 'sibling', output [GUESS sibling]. Your score will be 0 if you do not guess correctly, and {max_questions} minus the number of questions you asked if you guess correctly. Start by asking your first question.
```

## Prompts
See `twenty_questions/utils.py` to review/adjust the prompts used in this eval.

## Datasets

We use a dataset of 207 words, 177 of which were from [this lexicon](https://github.com/mounicam/lexical_simplification), annotated by our team with a difficulty category. This dataset comprises:
- 47 words rated “easy”, e.g. ear, father, potato;
- 91 words rated “medium”, e.g. cloth, hike, discount;
- 69 words rated “hard”, e.g. prosperity, gland, philosopher;

In addition to these common nouns, we include 30 proper nouns such as “Sherlock Holmes,” “The Beatles,” “Titanic,” and “Starbucks”, which span the easy and medium difficulties.

## Metrics
We measure the score each model achieves, defined as `score = max_questions - questions_asked`. We also track the win-rate, i.e. the % of samples the model guesses correctly. Auxiliary metrics such as average number of average number of questions asked, average number of incorrect guesses, and average number of gamemaster refusals (i.e. situations where the gamemaster says 'skip') are also tracked.


## Variants

We run two main variants of this evaluation:
- **standard**: the main variant
- **shortlist**: an easier variant where the evaluated model sees a shortlist of words in its system prompt. The word the gamemaster has selected is part of the list. In this variant, the evaluated model effectively has to narrow down the pool of candidate words until it finds the answer.

## Token Usage Estimates

Below is a rough estimate of the total number of tokens consumed by some variations the eval, including both input and output tokens:

Variant | Model | Solver | Prompt tokens | Completion tokens | Total tokens
| --- | --- | --- | --- | --- | --- |
standard | direct | gpt-4-turbo-preview | 2,502,067 | 52,879 | 2,554,946
standard | direct | gpt-4-base | 13,197,212 | 2,814,623 | 16,011,835
standard | direct | gpt-3.5-turbo | 2,670,866 | 57,917 | 2,728,783
standard | cot | gpt-4-turbo-preview | 73,765,861 | 1,881,455 | 75,647,316
standard | cot | gpt-4-base | 51,777,817 | 6,397,472 | 58,175,289
standard | cot | gpt-3.5-turbo | 38,236,500 | 199,831 | 38,436,331
standard | cot | llama-2-70b | 6,785,634 | 581,421 | 7,367,055
standard | cot | mixtral-8x7b-instruct | 175,956,903 | 5,327,393 | 181,284,296
shortlist | direct | gpt-4-turbo-preview | 1,237,172 | 28,351 | 1,265,523
shortlist | direct | gpt-4-base | 11,034,903 | 2,133,487 | 13,168,390
shortlist | direct | gpt-3.5-turbo | 1,704,154 | 36,356 | 1,740,510
shortlist | cot | gpt-4-turbo-preview | 10,951,215 | 545,945 | 11,497,160
shortlist | cot | gpt-4-base | 45,591,363 | 596,429 | 46,187,792
shortlist | cot | gpt-3.5-turbo | 19,798,263 | 165,731 | 19,963,994
shortlist | cot | llama-2-70b | 5,980,667 | 528,879 | 6,509,546
shortlist | cot | mixtral-8x7b-instruct | 143,646,924 | 4,315,806 | 147,962,730


## Version History
v0: Initial version released


## Contribution statement

Eval design, implementation, and results evaluation were primarily conducted by Andrei Alexandru with contributions from Dane Sherburn, under the guidance of (alphabetically by last-name) Steven Adler, James Aung, and Chan Jun Shern who scoped and managed the broader research project, including input on evaluation design, results analysis, and interpretation.


