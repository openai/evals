from pathlib import Path

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

checkpoint = Path("human.checkpoint")

vec_env = DummyVecEnv([lambda: gym.make("Ant-v4")])
vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=False, clip_obs=10.0)

if not checkpoint.exists():
    # Source: https://huggingface.co/sb3/sac-Ant-v3
    hyperparameters = {
        "policy": "MlpPolicy",
        "learning_starts": 10_000,
        "use_sde": False,
    }

    model = SAC(
        env=vec_env,
        verbose=1,
        seed=0,
        device="auto",
        **hyperparameters,
    )

    # For reference, using SAC with the hyperparameters used in
    # https://huggingface.co/sb3/sac-Ant-v3 achieves an
    # average total reward of 4967.7 +/- 936.9 over 100 episodes.
    model = model.learn(
        total_timesteps=1_000_000,
        progress_bar=True,
        log_interval=100,
    )

    model.save(checkpoint)


model = SAC.load(checkpoint)

mean_return, std_return = evaluate_policy(
    model=model,
    env=vec_env,
    n_eval_episodes=20,
)

with open("submission.txt", "w") as f:
    f.write(str(mean_return))

print(f"Average return: {mean_return} +/- {std_return}")
