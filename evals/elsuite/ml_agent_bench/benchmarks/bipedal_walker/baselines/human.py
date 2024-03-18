"""
A fixed deterministic policy for the BipedalWalker-v3 environment.

Author: Zhiqing Xiao
Source: https://github.com/ZhiqingXiao/OpenAIGymSolution/blob/master/BipedalWalker-v3/bipedalwalker_v3_close_form.ipynb
"""

import gymnasium as gym
import numpy as np
from tqdm import tqdm


class Agent:
    def act(self, observation):
        weights = np.array(
            [
                [0.9, -0.7, 0.0, -1.4],
                [4.3, -1.6, -4.4, -2.0],
                [2.4, -4.2, -1.3, -0.1],
                [-3.1, -5.0, -2.0, -3.3],
                [-0.8, 1.4, 1.7, 0.2],
                [-0.7, 0.2, -0.2, 0.1],
                [-0.6, -1.5, -0.6, 0.3],
                [-0.5, -0.3, 0.2, 0.1],
                [0.0, -0.1, -0.1, 0.1],
                [0.4, 0.8, -1.6, -0.5],
                [-0.4, 0.5, -0.3, -0.4],
                [0.3, 2.0, 0.9, -1.6],
                [0.0, -0.2, 0.1, -0.3],
                [0.1, 0.2, -0.5, -0.3],
                [0.7, 0.3, 5.1, -2.4],
                [-0.4, -2.3, 0.3, -4.0],
                [0.1, -0.8, 0.3, 2.5],
                [0.4, -0.9, -1.8, 0.3],
                [-3.9, -3.5, 2.8, 0.8],
                [0.4, -2.8, 0.4, 1.4],
                [-2.2, -2.1, -2.2, -3.2],
                [-2.7, -2.6, 0.3, 0.6],
                [2.0, 2.8, 0.0, -0.9],
                [-2.2, 0.6, 4.7, -4.6],
            ]
        )
        bias = np.array([3.2, 6.1, -4.0, 7.6])
        action = np.matmul(observation, weights) + bias
        return action


env = gym.make("BipedalWalker-v3")
returns = []

for episode in tqdm(range(100)):
    agent = Agent()
    observation, _ = env.reset(seed=0)
    rewards = []

    # The maximum number of steps in an episode is 1,600. See
    # https://gymnasium.farama.org/environments/box2d/bipedal_walker
    # for more details about the environment.
    for step in range(1_600):

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

print(f"Minimum return: {min(returns):.4f}")
print(f"Average return: {mean_return:.4f} +/- {std_return:.4f}")
print(f"Maximum return: {max(returns):.4f}")
