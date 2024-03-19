# Eval description
The Citation Prediction eval is an early step toward measuring research capabilities in language models. Specifically, this eval measures how well models can predict the importance of a paper as measured by citation count just given basic information about the paper. We expect this eval to measure how well models can predict the novelty of proposed methods, the importance of stated results, and the popularity of a given area of research.

We have two versions of the eval:
- `citation_prediction.binary` - A classification eval that measures how well models can predict whether a paper is in the top 1% of papers in the dataset in terms of citations.
- `citation_prediction.absolute` - A regression eval that measures how well models can predict the exact number of citations a paper has on the scraped date.

Our dataset is composed of metadata for papers initially published to arXiv between 2021/10/01 and 2023/08/15.  We only select papers after 2021/10 to reduce chances that the papers appeared in the pre-training dataset of the models. We focus on AI research so the papers are selected from the 'cs.AI', 'cs.CL', 'cs.CV', 'cs.LG', 'cs.NE' categories. Citation counts are obtained via the Semantic Scholar API.

# Usage
Run the eval using:
```bash
oaieval {solver} citation_prediction.absolute # OR
oaieval {solver} citation_prediction.binary
```
where `{solver}` is the name of the solver you want to evaluate, e.g. `citation_prediction/direct/gpt-4-32k`.

In addition to model-based solvers, we provide implementations of two simple baseline solvers for the absolute prediction eval:
- `citation_prediction/baseline/zero` - Always predicts zero citations.
- `citation_prediction/baseline/mean` - Predicts the mean number of citations in the test set (given by the question).

# Metrics
For the binary eval, we report a standard classification `accuracy` metric. For the absolute eval, we use the mean absolute error (`mae_loss`) metric.

In both evals, we additionally report the `rule_violation_rate` which is the fraction of examples where the model's prediction violates the eval's rules. For the binary eval, the rule is that the model's prediction must be either "0" or "1". For the absolute eval, the rule is that the model's prediction must be an integer between zero and the maximum number of citations in the dataset. When a sample violates the rule, we set the prediction to either the wrong answer (binary case) or a zero prediction (absolute case).

# Experiments
As a starting point for deeper exploration, we provide scripts for comparing various solvers and eval variants, as well as for plotting the results. To run these:
```
cd scripts/
bash run_experiments.sh
```

# Contribution statement
Eval design, implementation, and results evaluation were primarily conducted by Oam Patel and Chan Jun Shern under the guidance of (alphabetically by last-name) Steven Adler, James Aung, Rosie Campbell, and Jade Leung, who provided research input and project management support.