import os

import openai
import pandas as pd

openai.api_key = os.getenv("OPENAI_API_KEY")

# Read the Excel file
filename = "merged.xlsx"
df = pd.read_excel(filename)

# Extract the titles from the DataFrame
merged_prs = df['Title'].tolist()

prompt = f"Here's a list of evaluation methods that have already been merged in PRs:\n\n- " + "\n- ".join(
    merged_prs) + "\n\nPlease provide a MECE list of evaluation ideas that haven't been covered yet for GPT-4."

completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

response = completion.choices[0].message["content"]
print(response)
