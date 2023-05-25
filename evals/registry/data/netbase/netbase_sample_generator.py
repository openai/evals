#!/usr/bin/env python3
import netbase

system_prompt="You are an expert of common world knowledge akin to wikidata. Answer with an approximate number. Do not output any words"
template='{"input":[{"role":"system","content":"$prompt"},{"role":"user","content":"$user_prompt"}],"ideal":"$value"}'
template=template.replace("$prompt", system_prompt)
user_prompt="What is the population of $place?"

with open("new_samples.jsonl", "w") as file:
	for place, population in netbase.all.population:
		pop = int(population.value)
		if pop<=0:
			continue
		sample=template.replace("$user_prompt", user_prompt.replace("$place",place.name))
		print(sample.replace("$value", str(pop)))
		print(sample.replace("$value", str(pop)), file=file)
  
