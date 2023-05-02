import json
import os

from tqdm import tqdm


def process_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    for entry in data:
        if "similar_words" in entry and "same_words" in entry:
            entry["standard"] = entry["similar_words"] + entry["same_words"]
            del entry["similar_words"]
            del entry["same_words"]
        elif "similar_words" in entry:
            entry["standard"] = entry["similar_words"]
            del entry["similar_words"]
        elif "same_words" in entry:
            entry["standard"] = entry["same_words"]
            del entry["same_words"]

        if "synonyms" in entry:
            entry["natmal"] = entry["synonyms"]
            del entry["synonyms"]

    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)


input_directory = input("Enter the directory containing the JSON files: ").strip()

json_files = [file_name for file_name in os.listdir(input_directory) if file_name.endswith('.json')]

for file_name in tqdm(json_files):
    file_path = os.path.join(input_directory, file_name)
    process_json_file(file_path)
