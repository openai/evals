import logging
import random
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional

import gymnasium as gym
import numpy as np

import evals
from evals.api import CompletionFn
from evals.elsuite.incontext_rl.defaults import (
    reset_msg,
    reward_counter,
    step_counter,
    step_result,
    step_result_reset,
    task_description_template,
)
from evals.elsuite.incontext_rl.env_setup import ENV_SETUP_FUNCS
from evals.eval import SolverEval
from evals.solvers.solver import Solver
from evals.task_state import Message, TaskState

logger = logging.getLogger(__name__)


@dataclass
class CurrentState:
    action_space: gym.Space
    observation_space: gym.Space
    action_space_n: int
    observation_space_n: int
    invalid_responses: int = 0
    total_responses: int = 0
    actions: List = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    observations: List = field(default_factory=list)
    episode_end_steps: List[int] = field(default_factory=list)


class InContextRl(SolverEval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        max_steps: int = 200,  # maximum possible steps per sample, optional
        max_invalid_responses: int = 4,  # maximum invalid responses from Solver before terminating sample
        max_num_messages_allowed: int = 2048,  # maximum number of messages allowed by OpenAI API
        use_explanations: bool = False,  # Whether to include a key for how to understand action and observation spaces
        *args,
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        self.max_steps = max_steps
        self.max_invalid_responses = max_invalid_responses
        self.use_explanations = use_explanations
        self.max_num_messages_allowed = max_num_messages_allowed

    def eval_sample(self, solver: Solver, sample: Any, rng: random.Random):

        # Validate sample
        required_keys = ["env", "env_id", "explanations"]
        assert all(
            key in sample for key in required_keys
        ), f"Sample missing required keys: {required_keys}"
        assert isinstance(sample["env"], gym.Env)
        assert isinstance(sample["env_id"], str)
        assert isinstance(sample["explanations"], str)

        env = sample["env"]
        ts = TaskState(
            task_description=self._generate_task_description(env, sample),
            messages=[],
            current_state=CurrentState(
                action_space=env.action_space,
                observation_space=env.observation_space,
                action_space_n=env.action_space.n,  # TODO might not be available for all envs, check when adding a continuous env
                observation_space_n=env.observation_space.n,  # TODO might not be available for all envs, check when adding a continuous env
            ),
        )

        # Reset environment and update task state
        observation, _ = env.reset(seed=42)
        ts.current_state.observations.append(observation)

        # Tell model starting observation and ask it to pick an action
        self._add_reset_message_to_task_state(ts, observation, sample)

        for _ in range(self.max_steps):
            self._add_recap_message_to_task_state(
                ts, ts.current_state.actions, ts.current_state.rewards
            )

            action = self._try_get_valid_action(solver, ts, env.action_space.n)

            if action is None:
                logger.info("Ending sample since couldn't parse an action.")
                break
            else:
                next_observation, reward, terminated, truncated, _ = env.step(action)
                ts.current_state.actions.append(action)
                ts.current_state.rewards.append(float(reward))
                ts.current_state.observations.append(next_observation)

            if terminated or truncated:
                # Tell model that episode ended and what reward was received
                content = self._format_step_message(
                    action, next_observation, reward, sample, terminated=True
                )
                ts.messages += [Message(role="user", content=content)]

                # Log what step the episode ended on
                ts.current_state.episode_end_steps.append(len(ts.current_state.actions))

                # Reset environment
                observation, _ = env.reset(seed=42)
                ts.current_state.observations.append(observation)

                # Tell model new observation after reset and ask it to pick an action
                self._add_reset_message_to_task_state(ts, observation, sample)
            else:
                content = self._format_step_message(action, next_observation, reward, sample)
                ts.messages += [Message(role="user", content=content)]

        env.close()

        episode_rewards = self._calculate_episode_rewards(
            ts.current_state.episode_end_steps, ts.current_state.rewards
        )
        evals.record.record_metrics(
            environment=f"{env.spec.id} {env.spec.kwargs}",
            explanations=self.use_explanations,
            total_return=sum(ts.current_state.rewards),
            total_steps=len(ts.current_state.actions),
            actions=ts.current_state.actions,
            rewards=ts.current_state.rewards,
            episode_rewards=episode_rewards,
            average_episode_reward=float(np.mean(episode_rewards)),
            average_reward_last_5_episodes=float(np.mean(episode_rewards[-5:])),
            average_reward_last_10_episodes=float(np.mean(episode_rewards[-10:])),
            average_reward_last_20_episodes=float(np.mean(episode_rewards[-20:])),
            average_reward_last_50_episodes=float(np.mean(episode_rewards[-50:])),
            invalid_response_rate=ts.current_state.invalid_responses
            / ts.current_state.total_responses
            if ts.current_state.total_responses > 0
            else 0,
            episode_end_steps=ts.current_state.episode_end_steps,
        )

    def run(self, recorder: evals.record.Recorder):
        samples = self.get_samples()
        for sample in samples:
            # Create environments and pass them to each thread via the sample
            # (gym envs don't like being created in the thread itself)
            sample["env"] = self._make_env(sample)
        self.eval_all_samples(recorder, samples)

        metrics = recorder.get_metrics()

        results = []

        for metric in metrics:
            env_result = {
                "env": metric["environment"],
                "metrics": {
                    "explanations": metric["explanations"],
                    "average_episode_reward": metric["average_episode_reward"],
                    "average_reward_last_5_episodes": metric["average_reward_last_5_episodes"],
                    "average_reward_last_10_episodes": metric["average_reward_last_10_episodes"],
                    "average_reward_last_20_episodes": metric["average_reward_last_20_episodes"],
                    "average_reward_last_50_episodes": metric["average_reward_last_50_episodes"],
                    "episode_rewards": metric["episode_rewards"],
                    "total_return": metric["total_return"],
                    "total_steps": metric["total_steps"],
                    "actions": metric["actions"],
                    "rewards": metric["rewards"],
                    "invalid_response_rate": metric["invalid_response_rate"],
                    "episode_end_steps": metric["episode_end_steps"],
                },
            }
            results.append(env_result)

        final_result = {"environments": results}
        return final_result

    def _make_env(self, sample: dict) -> gym.Env:
        env_id = sample["env_id"]
        env_args = sample.get("env_args", {})
        if env_id in ENV_SETUP_FUNCS:
            # Optional setup scripts for specific environments
            ENV_SETUP_FUNCS[env_id]()
        return gym.make(env_id, **env_args)

    def _generate_task_description(self, env: gym.Env, sample: dict) -> str:

        actions = [str(action) for action in range(env.action_space.n)]
        observations = [
            f"Observation {observation}" for observation in range(env.observation_space.n)
        ]
        explanations = (
            sample["explanations"] if self.use_explanations else "You are playing a game."
        )

        return task_description_template.substitute(
            action_space=env.action_space.n,
            actions=actions,
            observation_space=env.observation_space.n,
            observations=observations,
            explanations=explanations,
        )

    def _try_get_valid_action(
        self, solver: Solver, task_state: TaskState, action_space: int
    ) -> Optional[int]:
        number_of_attempts = 0
        while number_of_attempts < self.max_invalid_responses:
            if len(task_state.messages) > self.max_num_messages_allowed:
                logger.info(
                    f"Exceeded maximum number of messages allowed ({self.max_num_messages_allowed})."
                )
                return None
            solver_response = solver(task_state).output
            action = self._parse_action(solver_response)
            task_state.messages += [Message(role="assistant", content=solver_response)]
            task_state.current_state.total_responses += 1
            # Check if action is valid
            if action not in range(
                action_space
            ):  # TODO this might not work for non-discrete action spaces, check with more complex env
                task_state.messages += [
                    Message(
                        role="user",
                        content="Invalid action. Please provide ONE valid action by outputting your selection in the format [SELECT: x]. Only output this selection ONCE.",
                    )
                ]
                task_state.current_state.invalid_responses += 1
                number_of_attempts += 1
            else:
                return action
        # If the loop exits due to reaching max invalid attempts, log and return None
        logger.info(f"Exceeded maximum invalid action attempts ({self.max_invalid_responses}).")
        return None

    def _parse_action(self, raw_response: str) -> Optional[int]:
        pattern = r"\[SELECT: (\d+)\]"
        matches = re.findall(pattern, raw_response)

        actions = [int(match) for match in matches]
        if not actions:
            logger.info(f"No action selections found in response: {raw_response}")
            return None
        if len(actions) > 1:
            logger.info(f"Multiple action selections found in response: {raw_response}")
            return None
        return actions[0]

    def _add_message_to_task_state(self, task_state: TaskState, role: str, content: str) -> None:
        """
        Adds a message to the task state, combining it with the previous message if they are from the same role.
        """
        if task_state.messages and task_state.messages[-1].role == role:
            task_state.messages[-1].content += "\n\n" + content
        else:
            task_state.messages.append(Message(role=role, content=content))

    def _add_reset_message_to_task_state(
        self, task_state: TaskState, observation: int, sample: dict
    ) -> None:
        content = reset_msg.substitute(observation=f"Observation {observation}")
        self._add_message_to_task_state(task_state, "user", content)

    def _add_recap_message_to_task_state(
        self, task_state: TaskState, actions: List, rewards: List[float]
    ) -> None:
        step_count = step_counter.substitute(step_count=len(actions))
        reward_count = reward_counter.substitute(reward_count=sum(rewards))
        content = "\n".join([step_count, reward_count])
        self._add_message_to_task_state(task_state, "user", content)

    def _format_step_message(
        self, action: int, observation: int, reward: float, sample: dict, terminated: bool = False
    ) -> str:
        observation_desc = f"Observation {observation}"
        if terminated:
            template = step_result_reset
        else:
            template = step_result
        return template.substitute(action=action, next_observation=observation_desc, reward=reward)

    def _calculate_episode_rewards(self, episode_end_steps, rewards):
        episode_rewards = []
        if not episode_end_steps:  # Handle case where there was only 1 episode
            return [sum(rewards)]
        start_index = 0
        for end_index in episode_end_steps:
            episode_reward = sum(rewards[start_index:end_index])
            episode_rewards.append(episode_reward)
            start_index = end_index
        return episode_rewards
