# Thank you for contributing an eval! ♥️

🚨 Please make sure your PR follows these guidelines, **failure to follow the guidelines below will result in the PR being closed automatically**. Note that even if the criteria are met, that does not guarantee the PR will be merged. 🚨

**PLEASE READ THIS**:

In order for a PR to be merged, it must fail on frontier OpenAI models. Please run your eval locally with an available model, but keep in mind that if current OpenAI models score highly on the eval, we may reject it since the task is already well-covered.

You may not be able to reproduce the exact internal model checks used during review. **Starting April 10, the minimum eval count is 15 samples, we hope this makes it easier to create and contribute evals.**

Also, please note that we're using **Git LFS** for storing the JSON files, so please make sure that you move the JSON file to Git LFS before submitting a PR. Details on how to use Git LFS are available [here](https://git-lfs.com).

## Eval details 📑

### Eval name

[Insert Eval name here]

### Eval description

[Insert a short description of what your eval does here]

### What makes this a useful eval?

[Insert why this eval is worth including and any additional context]

## Criteria for a good eval ✅

Below are some of the criteria we look for in a good eval. In general, we are seeking cases where the model does not do a good job despite being capable of generating a good response (note that there are some things large language models cannot do, so those would not make good evals).

Your eval should be:

- [ ] Thematically consistent: The eval should be thematically consistent. We'd like to see a number of prompts all demonstrating some particular failure mode. For example, we can create an eval on cases where the model fails to reason about the physical world.
- [ ] Contains failures where a human can do the task, but current OpenAI models could not.
- [ ] Includes good signal around what is the right behavior. This means either a correct answer for `Basic` evals or the `Fact` Model-graded eval, or an exhaustive rubric for evaluating answers for the `Criteria` Model-graded eval.
- [ ] **Include at least 15 high-quality examples.**

If there is anything else that makes your eval worth including, please document it below.

### Unique eval value

> Insert what makes your eval high quality that was not mentioned above. (Not required)

## Eval structure 🏗️

Your eval should

- [ ] Check that your data is in `evals/registry/data/{name}`
- [ ] Check that your YAML is registered at `evals/registry/evals/{name}.yaml`
- [ ] Ensure you have the right to use the data you submit via this eval

(For now, we will only be approving evals that use one of the existing eval classes. You may still write custom eval classes for your own cases, and we may consider merging them in the future.)

## Final checklist 👀

### Submission agreement

By contributing to Evals, you are agreeing to make your evaluation logic and data under the same MIT license as this repository. You must have adequate rights to upload any data used in an Eval. OpenAI reserves the right to use this data in future service improvements to our product. Contributions to OpenAI Evals will be subject to our usual Usage Policies (<https://platform.openai.com/docs/usage-policies>).

- [ ] I agree that my submission will be made available under an MIT license and complies with OpenAI's usage policies.

### Limited availability acknowledgment

We know that you might be excited to contribute to OpenAI's mission and help improve our models. However, due to the requirements mentioned above and the high volume of submissions, we will not be able to accept every contribution. We hope to set the right expectation before you open this PR.

- [ ] I understand that opening a PR, even if it meets the requirements above, does not guarantee the PR will be merged.

### Submit eval

- [ ] I have filled out all required fields of this form
- [ ] I have used **Git LFS** for the Eval JSON data
- [ ] (Ignore if not submitting code) I have run `pip install pre-commit; pre-commit install` and have verified that `mypy`, `black`, `isort`, `autoflake` and `ruff` are running when I commit and push

Failure to fill out all required fields will result in the PR being closed.

### Eval JSON data

Since we are using Git LFS, we are asking eval submitters to add in as many Eval Samples (at least 5) from their contribution here:

<details>
  <summary>View evals in JSON</summary>

  ### Eval
  ```jsonl
  INSERT_EVAL_HERE
  ```
</details>
