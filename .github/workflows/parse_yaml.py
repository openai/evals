import sys
import yaml

def get_first_key(file_path):
    with open(file_path, 'r') as yaml_file:
        content = yaml.safe_load(yaml_file)
        first_key = next(iter(content))
        return first_key

if __name__ == "__main__":
    yaml_file_path = sys.argv[1]
    print(get_first_key(yaml_file_path))
