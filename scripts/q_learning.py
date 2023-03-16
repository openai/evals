import numpy as np
import random

# Parameters
n_states = 10
n_episodes = 500
alpha = 0.1
gamma = 0.99
epsilon = 0.2
max_steps = 100

# Initialize Q table
Q = np.zeros((n_states, 2))

# Initialize random rewards for each state
state_rewards = np.random.rand(n_states)

# Define the environment
def step(state, action):
    if action == 0:  # Move left
        next_state = max(0, state - 1)
    elif action == 1:  # Move right
        next_state = min(n_states - 1, state + 1)
    reward = state_rewards[next_state]
    return next_state, reward

# Q-Learning
for episode in range(n_episodes):
    state = random.randint(0, n_states - 1)
    done = False

    while not done:
        # Choose an action using epsilon-greedy strategy
        if random.uniform(0, 1) < epsilon:
            action = random.choice([0, 1])
        else:
            action = np.argmax(Q[state])

        # Take the action and observe the next state and reward
        next_state, reward = step(state, action)

        # Update the Q table
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])

        # Update the state
        state = next_state

        # If the agent reaches the last state, the episode is done
        if state == n_states - 1:
            done = True

print("Q table:")
print(Q)

def get_best_action(state):
    return np.argmax(Q[state])

# Test the trained agent
def test_agent(start_state):
    state = start_state
    actions = []
    steps = 0
    while state != n_states - 1 and steps < max_steps:
        best_action = get_best_action(state)
        actions.append(best_action)
        state, _ = step(state, best_action)
        steps += 1
    return actions

# Test multiple scenarios
def test_multiple_scenarios(start_states):
    for start_state in start_states:
        actions = test_agent(start_state)
        print(f"Starting state: {start_state}, Best actions: {actions}")

# Example usage
start_states = [0, 3, 5, 7]
print("\nTesting the trained agent in multiple scenarios:")
test_multiple_scenarios(start_states)
