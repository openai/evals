import json
import os

json_files_folder = '../data/ch-kr-idioms'
output_file_name = 'population'  # Default file name for using all the data

# Ask the user whether to use all data or a random sample
response = input("Do you want to use a random sample? (y/n)")
if response.lower() == 'y':
    output_file_name = 'sample'

# Define the output file path
output_file_path = f'C:/Users/kjs/Documents/GitHub/evals/evals/registry/data/ch-kr-idioms/{output_file_name}.jsonl'

# List all JSON files in the folder
json_files = [f for f in os.listdir(json_files_folder) if f.endswith('.json')]

# Process all JSON files and merge their contents into a single JSONL file
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    for json_file in json_files:
        json_file_path = os.path.join(json_files_folder, json_file)

        # Load the JSON file
        with open(json_file_path, 'r', encoding='utf-8') as input_file:
            data = json.load(input_file)

        for item in data:
            # Remove hyphens from the idiom
            idiom = item['expression'].replace('-', '').replace('^', '')

            # Create a new JSON object with the input prompt and expected answer
            new_item = {
                'input': [
                    {
                        'role': 'system',
                        'content': '당신은 고사성어 전문가입니다.'
                    },
                    {
                        'role': 'user',
                        'content': f'다음과 같은 뜻 모두에 해당하는 고사성어 한 가지의 발음만 번호 없이 출력하십시오: "{item["meanings"]}"'
                    }
                ],
                'ideal': idiom
            }

            # Write the new JSON object to the output file
            output_file.write(json.dumps(new_item, ensure_ascii=False) + '\n')

# Open the output file
os.startfile(output_file_path)
