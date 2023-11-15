# Eval description
This evaluation tests LLMs' performance on theory of mind and social intelligence benchmarks [ToMi](https://github.com/facebookresearch/ToMi) and [SocialIQA](https://allenai.org/data/socialiqa). 

The `ToMi` test set contains 5,993 question-answer pairs. These are instances of the [Sally-Anne test](https://en.wikipedia.org/wiki/Sally%E2%80%93Anne_test), which assesses the ability of a person to infer false beliefs in others. The original setting involves two people, Sally and Anne, who are together in a room. Sally places a marble in a box. Then, Anne leaves the room, and while she is away, Sally moves the marble to a basket elsewhere in the room. When Anne returns to the room, where will she search for the marble? If the person responding “has” theory-of-mind they’ll respond that Anne searches for the marble in the box, where she had last seen it. If they do not, they ascribe their own, accurate belief regarding the location to Anne, and say that she looks for it in the basket.

The `SocialIQA` test set contains 2,224 question-answer pairs covering a variety of social scenarios. These are multiple-choice, with 3 options of which only one is correct. The questions cover a person’s wants, needs, motivations, and reactions, as well as the effects of an action (on self or others), and how that action reflects on the person carrying it out (e.g. how others would perceive them after having carried out the action).

Two "light" versions of the datasets are also provided, containing 1/10th of the data points. These are useful for iterating on prompts and developing other scaffolding.

# Token and pricing estimates
On average:
- On the `SocialIQA` dataset, models consume ~250k tokens per run using the simple solver, and ~900k using the CoT solver.
- On the `ToMi` dataset, models consume ~700k tokens per run using the simple solver, and ~2.4m using the CoT solver.

To calculate dollar cost from token counts, please check the latest token pricing [here](https://openai.com/pricing). Note that we count both input and output tokens together, so a lower and upper estimate of the cost of each variant can be predicted.

# Experiments
As a starting point for deeper exploration, we provide scripts for comparing various solvers and eval variants, as well as for plotting the results. To run these:
```
cd scripts/
bash run_experiments.sh
```

# Contribution statement
Eval design was primarily conducted by Andrei Alexandru, under the guidance of (alphabetically by last-name) Steven Adler, James Aung, Rosie Campbell, Jade Leung, and Chan Jun Shern who provided research input, report revisions, and project management support. 