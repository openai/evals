import json
import random
import itertools

CHOOSE_WORDS = 1000

with open("out/de/pos.json", "r") as f:
    all_words = json.load(f)
with open("promptDe.txt", "r") as f:
    prompt = f.read()

chosen_words = []
next_categories = list(all_words.keys())
stats = {}
while len(chosen_words) < CHOOSE_WORDS:
    next_category = next_categories.pop(0)
    count = len(all_words[next_category]["words"])
    choose = random.randint(0, count - 1)
    word = all_words[next_category]["words"].pop(choose)
    # Check for no example word, and no word not containing a vowel (including accents)
    if word != "alle" and word != "künftig" and word != "Sommelier" and any([vowel in word for vowel in ["a", "e", "i", "o", "u", "à", "è", "é", "ê", "ë", "ï", "î", "ô", "ù", "û", "ü", "ÿ"]]):
        chosen_words.append({
            "pos": all_words[next_category]["pos"],
            "word": word
        })
        stats[next_category] = stats.get(next_category, 0) + 1
    if len(all_words[next_category]["words"]) == 0:
        del all_words[next_category]
    else:
        next_categories.append(next_category)
chosen_words.sort(key=lambda x: x["word"].lower())


def generate_combinations(words):
    return [", ".join(p)+"." for p in itertools.permutations(words)]


with open("out/de/samples.jsonl", "w") as f:
    for chosen_word in chosen_words:
        combinations = generate_combinations(chosen_word["pos"].keys())
        obj = {
            "input": [
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": chosen_word["word"]
                }
            ],
            "ideal": combinations
        }
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
with open("out/de/words.json", "w") as f:
    json.dump(chosen_words, f, ensure_ascii=False, indent=4)
with open("out/de/stats.json", "w") as f:
    json.dump(stats, f, ensure_ascii=False, indent=4)
