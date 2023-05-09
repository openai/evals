import openai
import json

openai.api_key = "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"

responses = []
with open("C:/Users/donav/evals/evals/registry/data/adultery-state-laws/adultery.jsonl", "r") as file:
    for line in file:
        json_line = json.loads(line)
        input_messages = json_line["input"]
        input_text = "".join([message["content"] + "\n" for message in input_messages])
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=input_text,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        )
        responses.append(response.choices[0].text)

print(responses)




