import json
import os
import random

from tqdm import tqdm

json_files_folder = 'E:/data/chengyu'
output_file_name = 'population'  # Default file name for using all the data

# Ask the user whether to use all data or a random sample
response = input("Do you want to use a random sample? (y/n)")
if response.lower() == 'y':
    output_file_name = 'sample'
    max_samples = int(input("Enter the desired sample size: "))
else:
    max_samples = float('inf')

# Define the output file path

output_file_path = f'C:/Users/kjs/Documents/GitHub/evals/evals/registry/data/chengyu/{output_file_name}.jsonl'

# List all JSON files in the folder
json_files = [f for f in os.listdir(json_files_folder) if f.endswith('.json')]

# Concatenate all the data into a single list
data = []
for json_file in json_files:
    json_file_path = os.path.join(json_files_folder, json_file)

    with open(json_file_path, 'r', encoding='utf-8') as input_file:
        data.extend(json.load(input_file))

# Sample the data and write to the output file
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    if max_samples == float('inf'):
        num_items_to_sample = len(data)
    else:
        num_items_to_sample = min(max_samples, len(data))

    sampled_data = random.sample(data, num_items_to_sample)
    total_items = len(sampled_data)

    for item in tqdm(sampled_data, desc='Processing items', total=total_items, unit='item', mininterval=0.1):
        try:
            idiom = item['expression'].replace('-', '').replace('^', '')

            # Create a new JSON object with the input prompt and expected answer
            ideal_answers = [idiom]
            if 'standard' in item:
                ideal_answers += item['standard']
            new_item = {
                'input': [
                    {
                        'role': 'system',
                        'content': '당신은 고사성어 전문가입니다.'
                    },
                    {
                        'role': 'user',
                        'content': f'다음과 같은 뜻 모두에 해당하는 고사성어 한 가지의 한글 표기만 번호 없이 출력하십시오: "{item["meanings"]}"'
                    }
                ],
                'ideal': ideal_answers
            }

            # Write the new JSON object to the output file
            output_file.write(json.dumps(new_item, ensure_ascii=False) + '\n')
        except KeyError:
            print(f'Error: Key "meanings" not found in {json_file_path}\nItem: {item}')

    tqdm.write(f'Processed {total_items} items')
os.startfile(output_file_path)
