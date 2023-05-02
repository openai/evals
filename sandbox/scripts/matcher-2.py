import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from time import sleep
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Load the first set of JSON files
folder_1 = "E:/data/stdict/JSON_20230208"
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


search_url = "https://dict.idioms.moe.edu.tw/idiomList.jsp?idiom={}&qMd=0&qTp=1&qTp=2"


def is_chengyu_on_website(chengyu, max_retries=3, backoff_factor=0.5):
    query = search_url.format(quote(chengyu))
    retries = 0
    while retries <= max_retries:
        try:
            response = requests.get(query)
            soup = BeautifulSoup(response.content, "html.parser")
            result_count = soup.find("span", {"class": "SearchResult"})
            if result_count:
                count = int(re.search(r"\d+", result_count.text).group())
                return count > 0
            return False
        except requests.exceptions.RequestException as e:
            if retries < max_retries:
                sleep((2 ** retries) * backoff_factor)
                retries += 1
            else:
                print(f"Failed to fetch data for {chengyu} after {max_retries} retries. Error: {e}")
                return False


def process_entry(entry):
    if ("original_language_info" in entry["word_info"] and
            "original_language" in entry["word_info"]["original_language_info"][0] and
            is_chengyu_on_website(entry["word_info"]["original_language_info"][0]["original_language"])):
        return entry
    return None


output_folder = "E:/data/temp"
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

    # Update progress
    progress["last_processed_file"] = start_index + i + 1
    save_progress()
