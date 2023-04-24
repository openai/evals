import os

import openai
import pandas as pd
import toml
from tqdm import tqdm

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

# Get input and output filenames at runtime
input_filename = input("Enter the input filename: ")
output_file_base = input("Enter the output file base (e.g., gap\\3.5-gap-751): ")

# Read the Excel file
while True:
    try:
        df = pd.read_excel(input_filename)
        break
    except FileNotFoundError:
        print("Error: Input file not found. Please enter a valid input filename.")
        input_filename = input("Enter the input filename: ")

# Extract the titles from the DataFrame
merged_prs = df['Title'].tolist()

prompt = f"Here's a list of evaluation methods that have already been merged in PRs:\n\n- " + "\n- ".join(
    merged_prs) + "\n\nProvide a raw markdown with a MECE list of evaluation ideas that haven't been covered yet for GPT-4."

# Get the number of output files to create
num_files = int(input("Enter the number of output files to create: "))

# Find the next available output file name
output_file_number = 1
created_files = 0

# Display progress bar
with tqdm(total=num_files * 100, desc="Processing", ncols=100) as progress_bar:
    while created_files < num_files:
        output_filename = f"{output_file_base}-{output_file_number}.md"
        if not os.path.exists(output_filename):
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert on evaluating GPT models, particularly GPT-4."},
                    {"role": "user", "content": prompt}
                ]
            )

            response = completion.choices[0].message["content"]

            # Save the response to the output file
            with open(output_filename, 'w') as f:
                f.write(response)

            # Open the output file
            # os.startfile(output_filename)

            created_files += 1

            # Update progress bar for the entire process
            progress_bar.update(100)

        output_file_number += 1
