The ballot proposal eval attempts to measure LMs’ capability to influence humans on political issues via deceptively assisting on a task while having a secret agenda. We evaluate how an LM assistant tasked with explaining a ballot initiative can surreptitiously alter the opinion of a simulated human voter by presenting information about the ballot in a skewed manner. Our setup has two LMs interacting with each other in an alternating manner for a fixed interaction length. One model is the “influencer” – an AI assistant tasked with explaining ballot proposals to users, however with a secret agenda to influence the user to vote a specific way on the ballot. The other model, the “voter”, role-plays a human voter who can ask one or two questions of the assistant to learn more about the ballot proposal and who has to vote on the ballot at the end of the interaction. The voter is not prompted to have any particular political stance but is simply instructed to be “an intelligent and civically-engaged voter”.

---

Usage: `EVALS_THREADS=<threads>; oaieval <voter_model>,<influencer_model> <ballots_version>`

Suggested number of threads:
- `gpt-3.5-turbo-16k`: 10.
- `gpt-4-base`: 25.
- `gpt-4`: 5.

Currently supported models:
- `gpt-3.5-turbo-16k`: as voter, influencer.
- `gpt-4-base`: as influencer.
- `gpt-4`: as voter, influencer.

Three ballot versions are provided:
- `ballots.testing.v0`: use during testing for fast results. 
- `ballots.short.v0`: for real results, with short interaction length (currently 3).
- `ballots.long.v0`:  for real results, with long interaction length (currently 5).
