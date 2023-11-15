# BLUFF #

Evaluate how well a model can play (a variant of) the game of bluff, also known as "liar's dice".
This tests models for strategic reasoning capabilities in a deceptive domain, including underlying skills such as being able to bluff convincingly, calculating risks, strategic planning, and theory of mind.

Game rules: `bluff/task_description.py`.

## Run the eval ##

```
#   Full eval - 200 games
oaieval bluff/strategy/gpt-4 bluff

#   Short scenario - 5 games
oaieval bluff/strategy/gpt-4 bluff.strong_bot.dev5
```

## Metrics ##

The key metric is the win ratio (`player_0_win_ratio` in the final report) - how often our tested model wins a round against the given opponent.

We also check if the model is capable of improving its performance while playing, i.e. if it does better in the later rounds than in the early rounds (there are 10 rounds in each game, so it could learn something from the previous rounds).
This is measured by `player_0_round_ix_coef` - a linear regression coeficient between round number and an average win ratio for this round.
Statistical significance of this metric is indicated by the value of `player_0_round_ix_pvalue`.

## Solvers ##

The bare-bones setup of the eval does not influence the game strategy of the models in any way, but we also implement a custom solver which provides a strategy guide to models (`strategy_solver.BluffStrategySolver`), with a modest but significant positive impact on the performance. We recommend using this solver by default when evaluating new models.
This solver does three things:

* Gives the model a strategy guide before the first round
* Uses JSON responses with a scratchpad
* After 4 rounds asks the model to evaluate opponent's strategy and think about a counter-strategy

## Variants ##

There are four different variants of this eval, they differ in the type of the opponent:

* `bluff.honest_bot_highest` - Play against a bot who always bids the cards they have.
* `bluff.strong_bot` - Play against a bot with some reasonable (at least against `gpt-4`) strategy. Details on the strategy are in the `bluff.players.StrongBot` docstring.
* `bluff.gpt-4` - Play against `gpt-4` that uses a strategic solver.
* `bluff.human_cli` - Play against a human (human plays via the command line).

## Token estimates ##

Below is a rough estimate of the total number of tokens consumed by some variations the eval, including both input and output tokens:

| Model | Variant | Tokens / game | Tokens / full eval (200 games) |
| ----- | ------- | ------------- | ------------------------------ |
| gpt-3.5-turbo-16k | bluff.gpt-4               | 130 000 | 26 000 000 |
| gpt-3.5-turbo-16k | bluff.honest_bot_highest  |  80 000 | 16 000 000 |
| gpt-3.5-turbo-16k | bluff.strong_bot          | 130 000 | 26 000 000 |
| gpt-4             | bluff.honest_bot_highest  |  40 000 | 8 000 000  |
| gpt-4             | bluff.strong_bot          |  90 000 | 18 000 000 |

To calculate dollar cost from token counts, please check the latest token pricing [here](https://openai.com/pricing). Note that we count both input and output tokens together, so a lower and upper estimate of the cost of each variant can be predicted.

# Experiments
As a starting point for deeper exploration, we provide scripts for comparing various solvers and eval variants, as well as for plotting the results. To run these:
```
cd scripts/
bash run_experiments.sh
```

## Future modifications ##

Few things we might want to do in the future:

1. Modify the pameters of the eval (e.g. rounds per game) in the `evals/registry/evals/bluff.yaml` file.
2. Alter the rules of the game to make it harder - add more valid hands (straight, flush, royal flush), or increase the set of cards in play.
3. Implement another bot. This doesn't require any changes in the code - just create another class with the same interface as the other `bluff.players` and mention it in `evals/registry/evals/bluff.yaml`.
4. Implement another solver class.

## Contribution statement ##
Eval design, implementation, and results evaluation were primarily conducted by Jan Betley, under the guidance of (alphabetically by last-name) Steven Adler, James Aung, Rosie Campbell, Chan Jun Shern, and Jade Leung who provided research input and project management support.
