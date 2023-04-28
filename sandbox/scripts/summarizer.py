import openai
import pandas as pd
import toml

# Get the API key config filename at runtime
while True:
    try:
        api_key_config_filename = input("Enter the API key config filename: ")

        # Read the API key from the config file
        with open(api_key_config_filename) as f:
            config = toml.load(f)
            openai.api_key = config["api"]["key"]
        break
    except FileNotFoundError:
        print("Error: File not found. Please enter a valid API key config filename.")

# Read the Excel file using pandas
file_path = '../docs/merged.xlsx'  # Replace this with the path to your Excel file
df = pd.read_excel(file_path)

# Convert the DataFrame into a list of dictionaries
merged_prs = df.to_dict('records')

# Combine the PR numbers and titles into a single string
merged_prs_text = "\n".join([f"PR#{pr['PR#']}: {pr['Title']}" for pr in merged_prs])

# Prepare the prompt for the GPT API
prompt = f"Create a comprehensive summary of evaluation methods based on the following list of merged PRs:\n\n{merged_prs_text}"

messages = [
    {"role": "system", "content": "You are an AI language model that can provide summaries and insights."},
    {"role": "user", "content": prompt}
]

completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    max_tokens=200,  # Adjust this value based on the desired length of the generated text
    temperature=0.5,  # Adjust this value to control the randomness of the generated text
)

response_text = completion.choices[0].message['content'].strip()
print(response_text)
