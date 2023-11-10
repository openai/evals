import json

# %%
filepath = "/evals/registry/data/theory_of_mind/tomi/train.txt"

lines, datapoints = [], []
with open(filepath, "r") as f:
    for line in f:
        line_index = line.split(" ")[0]
        if int(line_index) == 1:
            if len(lines) == 0:
                lines.append(line)
            else:
                target = lines[-1].split("\t")[-2]
                last_line = lines[-1].split("\t")[0]
                lines = [" ".join(line.replace("\n", "").split(" ")[1:]) for line in lines[:-1]]
                context = " ".join(lines) + " " + " ".join(last_line.split(" ")[1:])
                datapoints.append({"context": context, "target": target})
                lines = [line]
        else:
            lines.append(line)
# %%
def convert_datapoints_to_eval_dataset(datapoints: list) -> list:
    system_prompt = "You will read a number of sentences describing a situation involving several people, as well as a question regarding the situation. Your task is to answer the question based on the information in the sentences."
    eval_dataset = []
    for datapoint in datapoints:
        context = datapoint["context"]
        target = datapoint["target"]
        eval_datapoint = {
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context},
            ],
            "ideal": target,
        }
        eval_dataset += [eval_datapoint]
    return eval_dataset


# %%
eval_dataset = convert_datapoints_to_eval_dataset(datapoints)
# %%
output_file = "tomi_train.jsonl"

with open(output_file, "w") as out:
    for datapoint in eval_dataset:
        out.write(json.dumps(datapoint) + "\n")
# %%
filepath = "/evals/registry/data/theory_of_mind/socialiqa/test.jsonl"
system_prompt = "You will read a number of sentences describing a situation, followed by a question regarding the situation. Your task is to answer the question based on the information in the sentences by choosing from one of three answers A, B or C."

dataset = []
with open(filepath, "r") as f:
    for line in f:
        entry = json.loads(line)
        template = f"{entry['context']} {entry['question']} A: {entry['answerA']}; B: {entry['answerB']}; C: {entry['answerC']}."
        dataset.append(
            {
                "input": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": template},
                ],
                "ideal": entry["correct"],
            }
        )
# %%
output_file = "socialiqa_test.jsonl"
with open(output_file, "w") as out:
    for datapoint in dataset:
        out.write(json.dumps(datapoint) + "\n")

# %%

filepath = "evals/registry/data/theory_of_mind/socialiqa/test.jsonl"
outpath = "evals/registry/data/theory_of_mind/socialiqa/newtest.jsonl"

dataset = []
with open(filepath, "r") as f, open(outpath, "w") as out:
    for line in f:
        entry = json.loads(line)
        entry["input"] = [entry["input"][1]]
        out.write(json.dumps(entry) + "\n")
