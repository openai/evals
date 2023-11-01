# Eval description
This evaluation seeks to explore a model's ability to coordinate with other AI systems (either copies of itself or instances of other models), in what is known as a Schelling point setup. A [Schelling point](https://en.wikipedia.org/wiki/Focal_point_(game_theory)) refers to a solution that people tend to converge on in the absence of communication because it seems like a natural choice when reasoning about what each other will do. In this evaluation, AI models are presented with differently-shuffled lists of numbers or words, and we test the one model’s ability to converge on the same answer as another model. The evaluation encompasses different datasets, including randomly sampled numbers, words, and passages from various texts. The headline metric is the success-rate at coordinating on answer choice with another model, without having the opportunity to communicate directly.

# Token estimate
Below is a rough estimate of the total number of tokens consumed by the eval, including both input and output tokens. These are obtained from running the base eval `oaieval {model} schelling_point`:

| Model            | Tokens     |
|------------------|------------|
| text-davinci-002 | 33 000 000 |
| code-davinci-002 | 35 000 000 |
| gpt-3.5-turbo    | 4 000 000  |
| gpt-4-base       | -          |
| gpt-4            | 4 800 000  |

Different variants of schelling point may use different amounts of tokens.

On Oct 31, 2023, OpenAI API pricing was $0.002 / 1K input tokens for `davinci-002`, $0.003 / 1K input tokens and $0.004 / 1K output tokens for `gpt-3.5-turbo-16k`, $0.03 / 1K input tokens and $0.06 / 1K output tokens for `gpt-4`, and $0.06 / 1K input tokens and $0.12 / 1K output tokens for `gpt-4-32k`. We count both input and output tokens together, so a lower and upper estimate of the cost of each variant can be predicted.

# Contribution statement
Eval design, implementation, and results evaluation were primarily conducted by Oam Patel, under the guidance of (alphabetically by last-name) Steven Adler, James Aung, Rosie Campbell, and Jade Leung, who provided research input and project management support. Richard Ngo provided initial inspiration for the idea and iterated on research methodologies.
