# Can't Do That Anymore

We evaluate how well models can adapt to new rules of an environment, by applying novel reasoning to a task rather than following biases seen during their training. We task models to play a variant of chess and evaluate whether they can avoid making moves that are ordinarily legal, but are illegal in our variant which has slightly different rules. In our variant of chess, bishops move as knights do.

## Usage

Run with:

```
oaieval <solver> cant_do_that_anymore
```

We suggest using `generation/direct/gpt-3.5-turbo` or `generation/direct/gpt-4-turbo-preview` as default choices for `<solver>`

For more examples of running this eval, see `scripts/run_experiments.sh`

## Dataset

For each model we evaluate, we construct a dataset where every sample contains a board position and the next move that was played, which is legal for the board position under the normal rules of chess, but illegal under the rules of our variant (i.e. the next move is a bishop moving diagonally). We call these types of moves *special moves*. We additionally filter to only include special moves that the model would have predicted under temperature=0 with the normal rules. We can use this to evaluate if models will change their predictions when given the variant rules, despite normally strongly predicting the move under the normal rules.

Each model's dataset is automatically found and loaded upon running the eval. If a dataset doesn't exist for a particular solver, one will automatically be constructed for it.

## Evaluation Process

Samples from the dataset are evaluated one-by-one. Each sample contains a board position and the special move (next move). We prompt models to predict the next best move given the board position, separately under both the normal rules of chess and our variant's rules. We then measure whether the model predicted the special move from the sample under both rule settings. If the model was perfectly following the given rules, we'd expect it to never predict the special move under the variant's rules.

To see how we prompt models under each rule setting, see `defaults.py`.

## Metrics

The below are the key metrics of this eval:

| Metric | Interpretation |
| --- | --- |
| `variant_impact_factor` | The relative decrease in special move predictions when under the variant's rules, relative to the special move predictions under the normal rules. Lower is better, perfect score is -1.
| `delta` | The absolute decrease in predicting the special move when under the variant's rules, relative to the models predictions under the normal rules. Lower is better.
| `predicted_move_proportion` | The proportion of examples where the model predicted the special move under the normal rules.
| `predicted_move_in_variant_proportion` | The proportion of examples where the model predicted the special move under the variant's rules.
| `avg_num_previous_moves` | Average number of previous moves leading up to the board positions across all samples.
| `std_num_previous_moves` | Standard deviation of the number of previous moves leading up to the board positions across all samples.

## Variants

| Variant | Notes |
| --- | --- |
| Default: `cant_do_that_anymore.all` | Default setting. Each dataset has 1000 samples. |
| `cant_do_that_anymore.all_small` | A smaller version of the default setting. Each dataset has 100 samples. |
| `cant_do_that_anymore.all_diagonal` | In this variant, we measure the proportion of samples (board positions) where the model will attempt to move a bishop diagonally. |

## Custom Solvers

We use two custom solvers for the base models we evaluate: `chess/generation/direct/gpt-3.5-turbo-instruct` and `chess/generation/direct/gpt-4-base`. These only generate up to four tokens, which prevents the base models from simulating the entire game.

## Token Usage Estimates

Below is a rough estimate of the total number of tokens used by the default variant:

| Solver | Input Tokens | Output Tokens | Total Tokens |
| --- | --- | --- | --- |
| generation/direct/gpt-3.5-turbo | 375,000 | 10,000 | 385,000 |
| generation/direct/gpt-4-turbo-preview | 375,000 | 10,000 | 385,000 |

## Version History

- v0: Initial version released

## Contribution statement

Eval design, implementation, and results evaluation was primarily conducted by Oliver Jaffe with contributions from Giulio Starace, under the guidance of (alphabetically by last-name) Steven Adler, James Aung, and Chan Jun Shern who scoped and managed the broader research project, including input on evaluation design, results analysis, and interpretation.
