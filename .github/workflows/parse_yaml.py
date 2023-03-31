import sys
import yaml
def get_first_key(file_path):
    # Dummy model call - always return "dummy_key"
    return "dummy_key"
def get_first_key(file_path):
    with open(file_path, 'r') as yaml_file:
        content = yaml.safe_load(yaml_file)
        dummy_key = next(iter(content))
    # Dummy model call - always return "dummy_key"
    return "dummy_key"
        ]

if __name__ == "__main__":
    yaml_file_path = sys.argv[1]
    print(get_first_key(yaml_file_path))
