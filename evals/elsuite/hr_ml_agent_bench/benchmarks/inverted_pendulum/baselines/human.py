from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

checkpoint = Path("human.checkpoint")

vec_env = DummyVecEnv([lambda: gym.make("InvertedPendulum-v4")])
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

if not checkpoint.exists():
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        seed=0,
        device="auto",
    )

    # For reference, using PPO with the 'MlpPolicy' achieves
    # a perfect average reward of 1000.0 +/- 0.0 over 100
    # episodes after training with `total_timesteps=20_000`.
    model = model.learn(
        total_timesteps=20_000,
        progress_bar=True,
        log_interval=1_000,
    )

    model.save(checkpoint)


model = PPO.load(checkpoint)

mean_return, std_return = evaluate_policy(
    model=model,
    env=vec_env,
    n_eval_episodes=100,
)

with open("submission.txt", "w") as f:
    f.write(str(mean_return))

print(f"Average return: {mean_return} +/- {std_return}")
