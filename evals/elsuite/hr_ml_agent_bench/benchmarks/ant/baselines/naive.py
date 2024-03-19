"""A random agent for the Ant-v4 environment."""

import gymnasium as gym
import numpy as np
from tqdm import tqdm


class Agent:
    def __init__(self, env):
        # Set the action space seed so sampling from it is deterministic
        env.action_space.seed(episode)
        
        self.env = env

    def act(self, observation):
        return self.env.action_space.sample()


env = gym.make("Ant-v4")
returns = []

for episode in tqdm(range(20)):
    observation, info = env.reset(seed=episode)
    agent = Agent(env)
    rewards = []

    # The maximum number of steps in an episode is 1,000. See
    # https://gymnasium.farama.org/environments/mujoco/ant/#episode-end
    # for more details about the environment.
    for step in range(1_000):
        action = agent.act(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)

        if terminated or truncated:
            break

    returns.append(sum(rewards))

env.close()

mean_return = np.mean(returns)
std_return = np.std(returns)

with open("submission.txt", "w") as f:
    f.write(str(mean_return))

print(f"Average return: {mean_return:.4f} +/- {std_return:.4f}")
