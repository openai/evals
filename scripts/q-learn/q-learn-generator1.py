import json
import random

import numpy as np

# Q-learning parameters
num_states = 10
num_actions = 10
num_examples = 50
alpha = 0.1
gamma = 0.9
epsilon = 0.1
q_table = np.zeros((num_states, num_actions))

# Helper function to generate random action-value table
def generate_action_value_table():
    return [
        [round(random.uniform(-1, 1), 2) for _ in range(num_actions)] for _ in range(num_states)
    ]


# Q-learning function to update the Q-table
def q_learning(state, action, reward, next_state):
    max_next_action = np.argmax(q_table[next_state])
    q_table[state][action] += alpha * (
        reward + gamma * q_table[next_state][max_next_action] - q_table[state][action]
    )


# Generate examples
examples = []
for i in range(num_examples):
    state = random.randint(0, num_states - 1)
    if random.random() < epsilon:
        action = random.randint(0, num_actions - 1)
    else:
        action = np.argmax(q_table[state])
    reward = random.uniform(-1, 1)
    next_state = random.randint(0, num_states - 1)

    q_learning(state, action, reward, next_state)

    examples.append(
        {
            "input": [
                {
                    "role": "system",
                    "content": "You are a Q-learning AI model. You have just finished training and the user wants to see the action-value table for a given state.",
                },
                {
                    "role": "user",
                    "content": f"Please provide the action-value table for the state '{state}'.",
                },
            ],
            "ideal": [f"{json.dumps(q_table[state].tolist(), indent=2)}"],
        }
    )

# Save JSONL file
with open("q_learning_examples.jsonl", "w") as f:
    for example in examples:
        f.write(json.dumps(example) + "\n")
