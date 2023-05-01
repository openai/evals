import json
import os
from datetime import datetime

import openai
from dotenv import load_dotenv


def read_and_process_text(file_name):
    with open(file_name, 'r') as f:
        text = f.read()
    escaped_text = json.dumps(text)
    clean_text = escaped_text.replace('  ', ' ')
    return clean_text


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY_2")

while True:
    version = input("Enter the GPT version: (3).5-turbo or (4): ")
    if version in ["3", "4"]:
        break
    else:
        print("Invalid input. Please enter '3' or '4'.")

model_name = "gpt-"
if version == "3":
    model_name += "3.5-turbo"
else:
    model_name += "4"

while True:
    temp = input("Enter the sampling temperature between 0 and 2: ")
    if temp.isnumeric() and 0 <= float(temp) <= 2:
        temp = float(temp)
        break
    else:
        print("Invalid input. Please enter a number between 0 and 2.")

while True:
    choice = input("\n(W)rite a message or (r)ead from a file: ")

    if choice.lower() == "w":
        prompt = input("Send a message: ")  # gimme what gpt you are, nothing else
    elif choice.lower() == "r":
        file_name = input("Enter the file name: ")
        prompt = read_and_process_text(file_name)
    else:
        print("Invalid option. Please type w or r.")
        continue

    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": ""  #
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=temp,
        stream=True
    )

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d")  # %H:%M:%S
    # print(formatted_time, prompt, sep=" ")

    first_chunk = True

    # for chunk in response:
    #     if "content" in chunk.choices[0].delta:
    #         print(chunk.choices[0].delta["content"], end="", flush=True)

    for chunk in response:
        if "content" in chunk.choices[0].delta:
            if first_chunk:
                print(f"{model_name}: {chunk.choices[0].delta['content']}", end="",
                      flush=True)
                first_chunk = False
            else:
                print(chunk.choices[0].delta["content"], end="", flush=True)
