import json
import os
import random
import re
import subprocess
import sys
from glob import glob


def create_jsonl_item(json_data):
    try:
        word = json_data["word_info"]["word"].replace('-하다', '').replace('-', '').replace('^', '')
        synonyms = [lexical_info["word"] for sense_info in
                    json_data["word_info"]["pos_info"][0]["comm_pattern_info"][0]["sense_info"] for lexical_info in
                    sense_info.get("lexical_info", [])]
        ideal_answers = [word] + synonyms
        ideal_answers = [answer.replace('-하다', '') for answer in ideal_answers]
        ideal_answers = [re.sub(r"[^가-힣]", "", answer) for answer in ideal_answers]
        meanings = json_data["word_info"]["pos_info"][0]["comm_pattern_info"][0]["sense_info"][0]["definition"]

        new_item = {
            "input": [
                {
                    "role": "system",
                    "content": "당신은 고사성어 전문가입니다."
                },
                {
                    "role": "user",
                    "content": f'다음 뜻에 해당하는 고사성어의 한글 표기만 출력하십시오: "{meanings}"'
                }
            ],
            "ideal": ideal_answers
        }
        return new_item
    except KeyError as e:
        print(f"Error processing JSON data: {e}")
        return None


def main():
    json_files = glob("E:/data/korean/chengyu/*.json")
    random.shuffle(json_files)
    jsonl_data = []

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as file:
            json_data = json.load(file)
            jsonl_item = create_jsonl_item(json_data)
            if jsonl_item:
                jsonl_data.append(jsonl_item)

    output_file = "C:/Users/kjs/Documents/GitHub/evals/evals/registry/data/chengyu/korean.jsonl"
    with open(output_file, "w", encoding="utf-8") as outfile:
        for item in jsonl_data:
            json.dump(item, outfile, ensure_ascii=False)
            outfile.write("\n")

    if sys.platform.startswith('darwin'):
        subprocess.call(('open', output_file))
    elif os.name == 'nt':  # For Windows
        os.startfile(output_file)
    elif os.name == 'posix':  # For Linux, Mac, etc.
        subprocess.call(('xdg-open', output_file))


if __name__ == "__main__":
    main()
