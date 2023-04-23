import os
import random
import re

import jsonlines
import openai
import toml
from tqdm.auto import tqdm

# Get the API key config filename at runtime
while True:
    try:
        api_key_config_filename = input("Enter the API key config filename: ")

        # Read the API key from the config file
        with open(api_key_config_filename) as f:
            config = toml.load(f)
            openai.api_key = config["api"]["key"]
        break
    except FileNotFoundError:
        print("Error: File not found. Please enter a valid API key config filename.")

# Get the messages/prompts config filename at runtime
while True:
    try:
        messages_config_filename = input("Enter the messages/prompts config filename: ")

        # Load the messages/prompts config
        with jsonlines.open(messages_config_filename) as f:
            messages_configs = [msg for msg in f]
        break
    except FileNotFoundError:
        print("Error: File not found. Please enter a valid messages/prompts config filename.")

# Sample size
num_sample = int(input("Enter the number of PRs to sample: "))

# Output directory
output_directory = input("Enter the output directory: ")

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Choose an input: (r)andom or (s)pecific
while True:
    choice = input("Choose an input: (r)andom or (s)pecific: ").lower()
    if choice in ['r', 's']:
        break
    else:
        print("Invalid input. Please enter 'r' or 's'.")

if choice == 'r':
    # Check if the requested sample size is larger than the dataset and handle accordingly
    if num_sample > len(messages_configs):
        print(
            f"Warning: The requested sample size ({num_sample}) is larger than the dataset size ({len(messages_configs)}). The entire dataset will be used.")
        sampled_messages_configs = messages_configs
    else:
        sampled_messages_configs = random.sample(messages_configs, num_sample)
else:
    indices = input(f"Enter the indices of the desired inputs separated by a space (0-{len(messages_configs) - 1}): ")
    selected_indices = list(map(int, indices.split()))
    sampled_messages_configs = [messages_configs[i] for i in selected_indices]


# Function to analyze evaluation tasks
def analyze_eval_task(preprocessed_text, messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
        # max_tokens=4097-,
    )

    return response.choices[0].message['content'].strip()


for messages_config in tqdm(sampled_messages_configs, desc="Processing PRs"):
    pr_text = messages_config["input"][-1]["content"]

    # Extract PR number from the pr_text
    pr_number_match = re.search(r'pull request (\d+):', pr_text)
    if pr_number_match:
        pr_number = pr_number_match.group(1)
    else:
        pr_number = 'unknown'

    # Analyze the PR text and generate insights
    analysis = analyze_eval_task(pr_text, messages_config["input"])

    # Save the analysis to a text file with the PR number included
    output_file = os.path.join(output_directory, f"pr-{pr_number}.txt")
    with open(output_file, "w", encoding='utf-8') as f:
        f.write(analysis)
