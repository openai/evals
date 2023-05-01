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


def generate_summary(conversation_history, model_name, temperature):
    summary_prompt = "Please provide a brief summary of the following conversation:\n\n"

    for message in conversation_history:
        role = message["role"].capitalize()
        content = message["content"]
        summary_prompt += f"{role}: {content}\n"

    summary_prompt += "\nSummary:"

    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "youre an expert on summarization"
            },
            {
                "role": "user",
                "content": summary_prompt
            }
        ],
        temperature=temp,
    )

    summary = response.choices[0]['message']['content'].strip()
    return summary


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

conversation_history = []

while True:
    choice = input("\n(W)rite a message or (r)ead from a file: ")

    if choice.lower() == "w":
        user_message = input("Send a message: ")
    elif choice.lower() == "r":
        file_name = input("Enter the file name: ")
        user_message = read_and_process_text(file_name)
    else:
        print("Invalid option. Please type w or r.")
        continue

    conversation_history.append({"role": "user", "content": user_message})

    summary = generate_summary(conversation_history, model_name, temp)

    context = [{"role": "system", "content": ""}, {"role": "assistant", "content": summary}]
    context.append({"role": "user", "content": user_message})

    response = openai.ChatCompletion.create(
        model=model_name,
        messages=context,
        temperature=temp,
        stream=True
    )

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

    first_chunk = True
    assistant_message = ""
    for chunk in response:
        if "content" in chunk.choices[0].delta:
            if first_chunk:
                assistant_message = chunk.choices[0].delta["content"]
                print(f"{model_name}: {assistant_message}", end="", flush=True)
                first_chunk = False
            else:
                print(chunk.choices[0].delta["content"], end="", flush=True)
                assistant_message += chunk.choices[0].delta["content"]

    conversation_history.append({"role": "assistant", "content": assistant_message})

    save_choice = input("\nDo you want to save this response to a file? (Y/N): ")
    if save_choice.lower() == "y":
        save_file_name = input("Enter the file name to save the response: ")
        with open(save_file_name, "w") as save_file:
            save_file.write(assistant_message)
        print(f"Response saved to {save_file_name}")
