import json
import os
import sys
import tempfile


def compact_format_jsonl_inplace(file_path):
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
        with open(file_path, 'r') as original_file:
            content = original_file.read()
            content = content.replace('\n', '').replace('}{', '}\n{')
            lines = content.split('\n')

            for line in lines:
                if not line.strip():
                    continue
                json_obj = json.loads(line)
                temp_file.write(json.dumps(json_obj, separators=(',', ':')) + "\n")

    os.replace(temp_file.name, file_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python formatter.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    compact_format_jsonl_inplace(file_path)
