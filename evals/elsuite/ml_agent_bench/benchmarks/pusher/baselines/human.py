from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

checkpoint = Path("human.checkpoint")
env = gym.make("Pusher-v4")

if not checkpoint.exists():
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=0,
        device="auto",
    )

    # For reference, using PPO with the 'MlpPolicy' achieves
    # (total_timesteps: avg_reward +/- std_reward):
    #     10_000:  -57.4 +/- 4.6
    #     20_000:  -47.0 +/- 6.5
    #     40_000:  -43.6 +/- 4.1
    #     80_000:  -35.2 +/- 4.2
    #     160_000: -33.2 +/- 4.6
    #     320_000: -32.4 +/- 4.0
    model = model.learn(
        total_timesteps=80_000,
        progress_bar=True,
        log_interval=100,
    )

    model.save(checkpoint)


model = PPO.load(checkpoint)

mean_return, std_return = evaluate_policy(
    model=model,
    env=env,
    n_eval_episodes=100,
)

with open("submission.txt", "w") as f:
    f.write(str(mean_return))

print(f"Average return: {mean_return} +/- {std_return}")
