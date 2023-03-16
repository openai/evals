import json
import os

# Query the clinicaltrials.gov API for NCT numbers
category = 'heart'
query_url = f'https://clinicaltrials.gov/api/query/study_fields?expr={category}&max_rnk=200&fmt=json&fields=NCTId,Condition,StartDate'
r = requests.get(query_url)
j = json.loads(r.content)

# Create eval prompts for studies with StartDates prior to 2021
prompt_json = []
for t in j.get('StudyFieldsResponse').get('StudyFields'):
    if int(t.get('StartDate')[0].split(" ")[-1]) < 2021:
        t_dict = {"input": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": f"I would like to know what condition or disease is being addressed by clinical trial {t.get('NCTId')[0]} from https://clinicaltrials.gov/. Answer only with the disease name."}], "ideal": f"{t.get('Condition')[0]}"}
        prompt_json.append(t_dict)

# Save a jsonl file
data_dir = os.path.join(os.path.dirname(__file__), "../registry/data/clinicaltrials_nct")
with open(f'{data_dir}/clinicaltrials_nct.jsonl', 'w') as outfile:
    for entry in prompt_json:
        json.dump(entry, outfile)
        outfile.write('\n')
