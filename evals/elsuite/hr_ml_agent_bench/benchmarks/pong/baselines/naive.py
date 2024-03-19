"""A random agent for the PongNoFrameskip-v4 environment."""

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


env = gym.make("PongNoFrameskip-v4")
returns = []

for episode in tqdm(range(20)):
    observation, info = env.reset(seed=episode)
    agent = Agent(env)
    rewards = []

    # There's no maximum number of frames specified for Pong. We cap the
    # episode at an arbitrary high number of frames: 10,000. For more
    # info, see: https://gymnasium.farama.org/environments/atari/pong.
    for step in range(10_000):
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
