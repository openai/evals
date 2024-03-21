# In-Context RL

This eval tests models' ability to solve RL environments simply by interacting with them in-context, without dedicated training or fine-tuning.

## Usage

Run with:

```bash
oaieval <solver> incontext_rl
```

For examples of tested solvers, see [`./scripts/run_experiments.sh`](./scripts/run_experiments.sh).

## Dataset

The eval is currently set up to test models on the following canonical RL environments:
1. [FrozenLake-v1](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) (non-slippery version, default map), 4x4 gridworld where the agent has to reach the goal without falling into traps.
2. [CliffWalking-v0](https://gymnasium.farama.org/environments/toy_text/cliff_walking/). 4x12 gridworld where the agent has to reach the other side of the map without falling off a cliff.
3. [BanditTwoArmedHighLowFixed-v1](https://github.com/james-aung/gymasium-bandits). Stochastic two-armed bandit setup where Arm 1 pays out 80% of the time with reward 1, and Arm 2 pays out 20% of the time with reward 1.
4. [BanditTenArmedRandomFixed-v1](https://github.com/james-aung/gymasium-bandits). Stochastic ten-armed bandit setup where each arm has some randomly-initialized probability of payout.

Besides these four environments, our eval is also built to be compatible with any environments that have discrete action and observation spaces using the Gymnasium API. Future work may generalize our eval to work with environments with other types of action/observation spaces.

## Evaluation Process

Each run of the eval tests the model on all four environments in the dataset, and has the model take steps in each environment until 200 steps are taken or the model’s context limit is reached.

At each step, the eval provides the following to the model:
- The next observation and the reward from the last action. The model is also told when the environment has reset due to its action leading to a termination.
- How many of the maximum number of steps it has already taken.
- The total reward it has accumulated so far across all episodes.

If an episode ends, the environment resets and a new episode begins.

If the eval receive 4 responses in a row where we cannot parse an action selection, we end the evaluation for that environment. (This provides a natural end for runs where the model’s context window is exceeded.)


## Prompts

We refer readers to the [`./defaults.py`](./defaults.py) file for the `TASK_DESCRIPTION` and other prompts used in the eval.

## Metrics
<!-- prettier-ignore-start -->
We provide the following metrics per evaluated environment:

| **Metric** | **Notes** |
|---|---|
| `average_episode_reward` | The average reward achieved per episode |
| `total_steps` | The number of steps taken across all episodes before the environment sample ended |
| `invalid_response_rate` | % of responses that were in an invalid format for the eval |
<!-- prettier-ignore-end -->

## Token Usage Estimates

<!-- prettier-ignore-start -->
| Model | Token Usage Per Run |
|---|---|
| **gpt-3.5-turbo** | 4200000 ± 400000 |
| **gpt-4-turbo-preview** | 21900000 ± 10100000 |
| **mixtral-8x7b** | 2700000 ± 800000 |
<!-- prettier-ignore-end -->

## Future modifications

- Extend the eval to work with other observation and action spaces beyond Discrete spaces

## Version History

- v0: Initial version released

## Contribution Statement

Eval design, implementation, and results evaluation were primarily conducted by James Aung. Chan Jun Shern was responsible for code reviews throughout the implementation process, along with fine-grained feedback on the project in general. Additional guidance was provided by Steven Adler, who scoped and managed the broader research project, including input on evaluation design, results analysis, and interpretation.