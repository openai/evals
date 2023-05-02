import os

import yaml

registry_path = "/evals/registry/evals"

for root, _, files in os.walk(registry_path):
    for file in files:
        if file.endswith('.yaml') or file.endswith('.yml'):
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    yaml.safe_load(f)
            except UnicodeDecodeError as e:
                print(f"Error in file: {file_path}")
                print(e)
