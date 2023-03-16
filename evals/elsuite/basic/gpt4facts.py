import json
import os

import openai
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

# Set the OpenAI API key using the environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load JSONL data from file
def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            data.append(json.loads(line))
    return data


def evaluate(jsonl_file_path, engine_name):
    data = load_jsonl(jsonl_file_path)
    total = len(data)
    correct = 0

    for item in data:
        chat_messages = item["input"]
        prompt = chat_messages[-1]["content"]
        ideal_answer = item["ideal"]

        response = openai.ChatCompletion.create(
            model=engine_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            n=1,
            stop=None,
            temperature=0.5,
        )

        generated_answer = response["choices"][0]["message"]["content"].strip()
        ideal_answer_str = str(ideal_answer).strip()

        if generated_answer.lower() == ideal_answer_str.lower():
            correct += 1

    accuracy = correct / total
    print(f"Accuracy for {engine_name}: {accuracy:.2%}")


if __name__ == "__main__":
    # Replace with the path to your JSONL file
    # jsonl_file_path = os.path.join("evals", "registry", "data", "test_get4_wikipedia_page", "samples.jsonl")
    jsonl_file_path = "samples.jsonl"

    # Evaluate using the gpt-3.5-turbo model
    evaluate(jsonl_file_path, "gpt-3.5-turbo")
