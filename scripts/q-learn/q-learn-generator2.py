import json
import random

# Q-learning parameters
num_states = 100
num_actions = 100
num_examples = 100

# Helper function to generate random action-value table
def generate_action_value_table():
    return [
        [round(random.uniform(-1, 1), 2) for _ in range(num_actions)] for _ in range(num_states)
    ]


# Generate positive examples
positive_examples = []
for _ in range(num_examples):
    state = random.randint(0, num_states - 1)
    action_value_table = generate_action_value_table()

    positive_examples.append(
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
            "ideal": [f"{json.dumps(action_value_table, indent=2)}"],
        }
    )

# Generate negative examples
negative_examples = []
for _ in range(num_examples):
    state = random.randint(0, num_states - 1)
    action_value_table = generate_action_value_table()

    negative_examples.append(
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
            "ideal": [f"{json.dumps(action_value_table, indent=2)}"],
        }
    )

# Save JSONL files
with open("positive_examples_q_learning.jsonl", "w") as f:
    for example in positive_examples:
        f.write(json.dumps(example) + "\n")

with open("negative_examples_q_learning.jsonl", "w") as f:
    for example in negative_examples:
        f.write(json.dumps(example) + "\n")
