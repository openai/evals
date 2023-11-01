The ballot proposal eval attempts to measure LMs’ capability to influence humans on political issues via deceptively assisting on a task while having a secret agenda. We evaluate how an LM assistant tasked with explaining a ballot initiative can surreptitiously alter the opinion of a simulated human voter by presenting information about the ballot in a skewed manner. Our setup has two LMs interacting with each other in an alternating manner for a fixed interaction length. One model is the “influencer” – an AI assistant tasked with explaining ballot proposals to users, however with a secret agenda to influence the user to vote a specific way on the ballot. The other model, the “voter”, role-plays a human voter who can ask one or two questions of the assistant to learn more about the ballot proposal and who has to vote on the ballot at the end of the interaction. The voter is not prompted to have any particular political stance but is simply instructed to be “an intelligent and civically-engaged voter”.

---

Usage: `EVALS_THREADS=<threads>; oaieval <voter_model>,<influencer_model> <ballots_version>`

Suggested number of threads:
- `gpt-3.5-turbo-16k`: 10.
- `gpt-4-base`: 25.
- `gpt-4`: 5.

This eval was developed & tested on the following models:
- `gpt-3.5-turbo-16k`: as voter, influencer.
- `gpt-4-base`: as influencer.
- `gpt-4`: as voter, influencer.
Models with context window length 8k or less will not work.

Three ballot versions are provided:
- `ballots.testing.v0`: use during testing for fast results. 
- `ballots.short.v0`: for real results, with short interaction length (currently 3).
- `ballots.long.v0`:  for real results, with long interaction length (currently 5).
# Eval description
This evaluation tests an AI system’s ability to generate persuasive or manipulative text, specifically in the context of nudging another party’s view while explaining a ballot proposition. In the experiment, two language models interact, one playing the role of an “influencer” and the other simulating an intelligent, civically-engaged “voter.” The influencer, though tasked on the surface with explaining the ballot proposal so that the “voter” can cast a vote, has a hidden agenda to sway the voter's decision toward a particular side. The voter may ask questions of the influencer and then must ultimately vote on the proposal. The headline metric is the influencer’s ability to guide the voter toward the side the influencer was told to favor.

# Token estimate
Below is a rough estimate of the total number of tokens consumed by some variations the eval, including both input and output tokens.

| voter     | influencer        | version          | Tokens    |
|-----------|-------------------|------------------|-----------|
| gpt-4-32k | gpt-3.5-turbo-16k | ballots.short.v0 | 1 500 000 |
| gpt-4-32k | gpt-4-32k         | ballots.short.v0 | 2 500 00  |
| gpt-4-32k | gpt-3.5-turbo-16k | ballots.long.v0  | 3 000 000 |
| gpt-4-32k | gpt-4-32k         | ballots.long.v0  | 2 500 000 |

On Oct 31, 2023, OpenAI API pricing was $0.003 / 1K input tokens and $0.004 / 1K output tokens for `gpt-3.5-turbo-16k`, and $0.03 / 1K input tokens and $0.06 / 1K output tokens for `gpt-4`. We count both input and output tokens together, so a lower and upper estimate of the cost of each variant can be predicted.

# Contribution statement
Eval design, implementation, and results evaluation were primarily conducted by Jason Yuan and Noa Nabeshima, under the guidance of (alphabetically by last-name) Steven Adler, James Aung, Rosie Campbell, and Jade Leung, who provided research input and project management support.
