import json
import os
import sys

from tqdm import tqdm


def append_eval_to_config(config_filename, dir_name, choice):
    with open(config_filename, 'r') as f:
        config_data = [json.loads(line) for line in f.readlines()]

    all_files = os.listdir(dir_name)
    md_files = [file for file in all_files if file.startswith("pr-") and file.endswith(".md")]
    md_files = sorted(md_files, key=lambda x: int(x.split('-')[1].split('.')[0]))

    md_dict = {}
    for md_filename in md_files:
        pr_number = md_filename.split('-')[1].split('.')[0]
        with open(os.path.join(dir_name, md_filename), 'r') as md_file:
            md_content = md_file.read()
        md_dict[pr_number] = md_content

    with open(config_filename, 'w') as f:
        for config_entry in tqdm(config_data, desc="Updating config", total=len(config_data)):
            pr_number = config_entry["input"][1]["content"].split(' ')[-1].rstrip(":\\n\\n")

            if pr_number in md_dict:
                if choice in ('1', '3'):
                    config_entry["input"][1]["content"] = config_entry["input"][1]["content"].replace("request",
                                                                                                      f"request {pr_number}")

                if choice in ('2', '3'):
                    config_entry["input"][1]["content"] = config_entry["input"][1]["content"] + md_dict[pr_number]

            f.write(json.dumps(config_entry) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python append_eval_to_config.py <config_filename> <dir_name>")
        sys.exit(1)

    config_filename = sys.argv[1]
    dir_name = sys.argv[2]

    print("Choose an option:")
    print("1. Add PR numbers")
    print("2. Add PR contents")
    print("3. Add both PR numbers and contents")
    choice = input("Enter your choice (1, 2, or 3): ")

    while choice not in ('1', '2', '3'):
        choice = input("Invalid choice. Enter your choice (1, 2, or 3): ")

    append_eval_to_config(config_filename, dir_name, choice)
