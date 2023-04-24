import os
import random
import threading

import openai
import pandas as pd
import toml
from tqdm import tqdm

# Read the API key from config.toml
with open("config.toml") as f:
    config = toml.load(f)
    openai.api_key = config["api"]["key"]


# Function to get the directory name at runtime with retries
def get_directory_name(prompt):
    while True:
        directory = input(prompt)
        if os.path.isdir(directory):
            return directory
        else:
            print(f"Error: '{directory}' is not a valid directory. Please try again.")


def extract_title(summary_text, timeout=30):
    result = {"title": "Title not found"}

    def worker():
        nonlocal result

        prompt = f"Read the following summary:\n\n{summary_text}\n\nCome up with a concise and informative title that captures the essence of the evaluation."
        messages = [
            {"role": "system", "content": "You are an expert on evaluating GPTs."},
            {"role": "user", "content": prompt}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0,
        )

        response_text = response.choices[0].message['content'].strip()

        title = response_text
        result["title"] = title.strip()

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        print("Skipping the current file due to timeout.")
        thread.join()

    return result["title"]


# Get the output directory name at runtime
output_directory = get_directory_name("Enter the directory name containing the summary files: ")

# Get the list of summary files in the output directory
summary_files = [os.path.join(output_directory, f) for f in os.listdir(output_directory) if
                 f.endswith(".txt")]

# Set test_mode to True for test mode, and False for real mode
test_mode = input("Enable test mode? (y/n): ").lower() == 'y'

if test_mode:
    while True:
        pr_numbers_to_process = input(
            "Enter the PR numbers of the files to process in test mode, separated by spaces (e.g., '1 2 3 4'): ")
        pr_numbers_to_process = pr_numbers_to_process.split()

        if all(x.isdigit() for x in pr_numbers_to_process):
            pr_numbers_to_process = [int(x) for x in pr_numbers_to_process]
            break
        else:
            print("Error: Invalid input. Please enter only space-separated numbers.")

    summary_files = [os.path.join(output_directory, f"pr-{pr_number}.txt") for pr_number in
                     pr_numbers_to_process]

# Initialize lists for PR numbers, titles
pr_numbers = []
titles = []

# Iterate through the summary files and extract PR numbers, titles with a progress bar
for summary_file in tqdm(summary_files, desc="Extracting titles", unit="file"):
    with open(summary_file, encoding='utf-8') as f:
        summary_text = f.read()

    # Extract the PR number from the file name
    pr_number = os.path.basename(summary_file).rstrip('.txt').split('-')[-1]

    # Extract the title from the summary text
    title = extract_title(summary_text)

    # Add the PR number, title to the lists
    pr_numbers.append(pr_number.strip())
    titles.append(title.strip())

# Create a DataFrame with columns for PR numbers, titles
data = {"PR#": pr_numbers, "Title": titles}
df = pd.DataFrame(data)

# Save the DataFrame as an Excel file
while True:
    output_filename = input("Enter the desired name for the output Excel file (e.g., 'output.xlsx'): ")

    if not output_filename.endswith('.xlsx'):
        print("Error: The filename must have an '.xlsx' extension.")
        continue

    try:
        df.to_excel(output_filename, index=False)
        print(f"File saved as '{output_filename}'.")
        os.startfile(output_filename)
        break
    except ValueError as e:
        print(f"Error: {e}")
        print("Please try again with a valid file name.")
    except PermissionError:
        print(
            f"Error: Permission denied when trying to save '{output_filename}'. Please make sure the file is not open or in use by another program.")
        user_choice = input("Press 'r' to retry, or 'q' to exit: ").lower()
        if user_choice == 'q':
            break
