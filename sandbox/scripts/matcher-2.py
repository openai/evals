import json
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm import tqdm

# Load the first set of JSON files
folder_1 = "E:/data/korean/JSON_20230208"
json_files_1 = [os.path.join(folder_1, file) for file in os.listdir(folder_1) if file.endswith(".json")]

# Read progress file, if available
progress_file = "progress.json"
if os.path.exists(progress_file):
    with open(progress_file, "r", encoding="utf-8") as f:
        progress = json.load(f)
else:
    progress = {}

# Continue from the last processed file
start_index = progress.get("last_processed_file", 0)


# Function to save progress
def save_progress():
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


# Read the Excel file with idioms
idioms_file = "E:\data\dict_idioms_2020_20230329.xls"
idioms_df = pd.read_excel(idioms_file, engine="xlrd")
idioms_set = set(idioms_df["成語"])


def is_chengyu_in_idioms(chengyu):
    return chengyu in idioms_set


def process_entry(entry):
    if ("original_language_info" in entry["word_info"] and
            "original_language" in entry["word_info"]["original_language_info"][0] and
            is_chengyu_in_idioms(entry["word_info"]["original_language_info"][0]["original_language"])):
        return entry
    return None


output_folder = "E:/data/chengyu"
os.makedirs(output_folder, exist_ok=True)

for i, file in enumerate(tqdm(json_files_1[start_index:], desc="Processing files", position=0)):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
        with ThreadPoolExecutor() as executor:
            matching_entries = list(tqdm(executor.map(process_entry, data["channel"]["item"]),
                                         total=len(data["channel"]["item"]),
                                         desc="Processing JSON items", position=1, leave=False))
            matching_entries = [entry for entry in matching_entries if entry is not None]

    for entry in matching_entries:
        output_file = os.path.join(output_folder, f"{entry['word_info']['word']}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2, ensure_ascii=False)
        tqdm.write(f"Written entry: {entry['word_info']['word']}")

    # Update progress
    progress["last_processed_file"] = start_index + i + 1
    save_progress()
