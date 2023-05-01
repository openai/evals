import json
import os


# Define a function to remove duplicates
def remove_duplicates(json_data):
    data = json.loads(json_data)

    for entry in data:
        if 'similar_words' in entry:
            entry['similar_words'] = list(set(entry['similar_words']))

    cleaned_json_data = json.dumps(data, ensure_ascii=False, indent=4)
    return cleaned_json_data


# Specify the directory containing JSON files
json_directory = 'C:/Users/kjs/Documents/GitHub/evals/sandbox/data/test'

# Iterate through all JSON files in the directory
for file_name in os.listdir(json_directory):
    if file_name.endswith('.json'):
        file_path = os.path.join(json_directory, file_name)

        # Read the contents of the JSON file
        with open(file_path, 'r', encoding='utf-8') as file:
            json_data = file.read()

        # Remove duplicates from the JSON data
        cleaned_json_data = remove_duplicates(json_data)

        # Write the cleaned JSON data back to the file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(cleaned_json_data)
