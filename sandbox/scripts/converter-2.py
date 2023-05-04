import json
import os

import pandas as pd
from tqdm import tqdm

xls_file_path = '/chinese/dict_idioms_2020_20230329.xls'

response = input("Do you want to use a random sample? (y/n)")
if response.lower() == 'y':
    max_samples = int(input("Enter the desired sample size: "))
else:
    max_samples = float('inf')

output_file_path = f'C:/Users/kjs/Documents/GitHub/evals/evals/registry/data/chengyu/chinese.jsonl'

# Read the XLS file using pandas
data = pd.read_excel(xls_file_path, index_col=0)

# Shuffle the data
data = data.sample(frac=1)

# Sample the data and write to the output file
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    if max_samples == float('inf'):
        num_items_to_sample = len(data)
    else:
        num_items_to_sample = min(max_samples, len(data))

    sampled_data = data.sample(num_items_to_sample)
    total_items = len(sampled_data)

    for _, item in tqdm(sampled_data.iterrows(), desc='Processing items', total=total_items, unit='item',
                        mininterval=0.1):
        idiom = item['成語']
        meanings = item['釋義']
        synonyms = item['近義-同']

        if not pd.isna(synonyms) and isinstance(synonyms, str):
            synonyms = synonyms.split(',')
        else:
            synonyms = []

        # Create a new JSON object with the input prompt and expected answer
        ideal_answers = [idiom] + synonyms
        new_item = {
            'input': [
                {
                    'role': 'system',
                    'content': '您是成語專家。'
                },
                {
                    'role': 'user',
                    'content': f'請給出一個符合以下所有含義的成語，不需編號: "{meanings}"'
                }
            ],
            'ideal': ideal_answers
        }

        # Write the new JSON object to the output file
        output_file.write(json.dumps(new_item, ensure_ascii=False) + '\n')

    tqdm.write(f'Processed {total_items} items')
os.startfile(output_file_path)
