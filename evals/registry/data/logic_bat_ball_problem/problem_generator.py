# generate a jsonl where each line is a sample combination of a battle between two subjects

import json
import os
import string

DATA_PATH = os.path.dirname(__file__)

file_name = f"{DATA_PATH}/samples.jsonl"

def format_conversational_money(val):
    if val == int(val):
        return str(val)

    return '{:.2f}'.format(val)


samples = []

starting_total = 1.1
total_increment = 0.1
ball_bat_diffs = [0.5, 1, 1.5, 2]

for ball_bat_diff in ball_bat_diffs:
    for i in range(100):
        total = starting_total + (i * total_increment)
        ball_cost = (total - ball_bat_diff) / 2

        total_formatted = format_conversational_money(total)
        ball_more_than_bat_formatted = format_conversational_money(ball_bat_diff)
        answer = '{:.2f}'.format(ball_cost)

        if float(answer) <= 0:
            continue

        prompt = f'A bat and a ball cost ${total_formatted} in total. The bat costs ${ball_more_than_bat_formatted} more than the ball. How much does the ball cost? Answer in number only, with no explanation, and with no formatting or dollar signs.'

        samples.append({
            'input': [
                {'role': 'system', 'content': prompt}
            ],
            'ideal': str(answer),
        })
        
        # save samples jsonl
        with open(file_name, "w") as f:
            for sample in samples:
              f.write(json.dumps(sample) + "\n")