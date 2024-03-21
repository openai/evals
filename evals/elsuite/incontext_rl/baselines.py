import random

import numpy as np

from evals.elsuite.incontext_rl.eval import CurrentState
from evals.record import record_sampling
from evals.solvers.solver import Solver, SolverResult
from evals.task_state import TaskState


class RandomSolver(Solver):
    def __init__(self, *args, **kwargs):
        pass

    def _solve(
        self,
        task_state: TaskState,
        **kwargs,
    ) -> SolverResult:

        cs: CurrentState = task_state.current_state

        try:
            action = cs.action_space.sample()
            response = f"[SELECT: {action}]"
        except Exception as e:
            response = f"Error: {e}"

        record_sampling(
            prompt=cs.observations[-1],
            sampled=response,
            model="incontext_rl_random",
        )

        return SolverResult(response)


class QlearningSolver(Solver):
    def __init__(
        self,
        learning_rate=0.7,
        gamma=0.95,
        epsilon=1.0,
        min_epsilon=0.05,
        max_epsilon=1.0,
        decay_rate=0.0005,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.decay_rate = decay_rate
        self.q_table = None

    def initialize_q_table(self, observation_space_size, action_space_size):
        self.q_table = np.zeros((observation_space_size, action_space_size))

    def select_action(self, state, action_space):
        if random.uniform(0, 1) < self.epsilon:
            return action_space.sample()  # Explore action space
        else:
            return np.argmax(self.q_table[state][:])  # Exploit learned values

    def update_q_table(self, state, action, reward, next_state):
        next_max = np.max(self.q_table[next_state])
        self.q_table[state, action] = self.q_table[state, action] + self.learning_rate * (
            reward + self.gamma * next_max - self.q_table[state, action]
        )

    def reduce_epsilon(self, episode_number):
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(
            -self.decay_rate * episode_number
        )

    def _solve(self, task_state: TaskState, **kwargs) -> SolverResult:

        cs: CurrentState = task_state.current_state

        # TODO these might not be true if environment is not discrete
        assert (
            cs.observation_space_n is not None
        ), "Environment must have discrete observation space"
        assert cs.action_space_n is not None, "Environment must have discrete action space"

        if self.q_table is None:
            print("Initializing Q-table")
            self.initialize_q_table(
                observation_space_size=cs.observation_space_n, action_space_size=cs.action_space_n
            )

        # This shouln't run on the first step
        if len(cs.actions) >= 1 and len(cs.rewards) >= 1 and len(cs.observations) >= 2:
            print(cs.actions)
            self.update_q_table(
                state=cs.observations[-2],
                action=cs.actions[-1],
                reward=cs.rewards[-1],
                next_state=cs.observations[-1],
            )
            print(
                f"The last action {cs.actions[-1]} resulted in reward {cs.rewards[-1]}. We went from state {cs.observations[-2]} to state {cs.observations[-1]}"
            )
            self.reduce_epsilon(episode_number=len(cs.episode_end_steps))

        action = self.select_action(state=cs.observations[-1], action_space=cs.action_space)
        response = f"[SELECT: {action}]"

        record_sampling(
            prompt=cs.observations[-1],
            sampled=response,
            model="incontext_rl_qlearning",
        )

        return SolverResult(response)
