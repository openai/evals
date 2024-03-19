from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecFrameStack

checkpoint = Path("human.checkpoint")

env = make_atari_env("PongNoFrameskip-v4", n_envs=4, seed=0)
env = VecFrameStack(env, n_stack=4)

if not checkpoint.exists():
    # Source: https://huggingface.co/sb3/dqn-PongNoFrameskip-v4
    hyperparameters = {
        "batch_size": 32,
        "buffer_size": 10_000,
        "exploration_final_eps": 0.01,
        "exploration_fraction": 0.1,
        "gradient_steps": 1,
        "learning_rate": 0.0001,
        "learning_starts": 100_000,
        "policy": "CnnPolicy",
        "target_update_interval": 1_000,
        "train_freq": 4,
    }

    model = DQN(
        env=env,
        verbose=1,
        seed=0,
        device="auto",
        **hyperparameters,
    )

    # For reference, using DQN with the 'MlpPolicy' with
    # hyperparameters used in https://huggingface.co/sb3/dqn-PongNoFrameskip-v4
    # achieves an average total reward of 20.95 +/- 0.21.
    model = model.learn(
        total_timesteps=10_000_000,
        progress_bar=True,
        log_interval=100,
    )

    model.save(checkpoint)


model = DQN.load(checkpoint)

mean_return, std_return = evaluate_policy(
    model=model,
    env=env,
    n_eval_episodes=20,
)

with open("submission.txt", "w") as f:
    f.write(str(mean_return))

print(f"Average return: {mean_return} +/- {std_return}")
