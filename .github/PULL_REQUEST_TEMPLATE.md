# Thank you for contributing an eval! ♥️

🚨 Please make sure your PR follows these guidelines, __failure to follow the guidelines below will result in the PR being closed automatically__. Note that even if the criteria are met, that does not guarantee the PR will be merged nor GPT-4 access granted. 🚨

__PLEASE READ THIS__:

In order for a PR to be merged, it must fail on GPT-4. We are aware that right now, users do not have access, so you will not be able to tell if the eval fails or not. Please run your eval with GPT-3.5-Turbo, but keep in mind as we run the eval, if GPT-4 gets higher than 90% on the eval, we will likely reject since GPT-4 is already capable of completing the task.

We plan to roll out a way for users submitting evals to see the eval performance on GPT-4 soon. Stay tuned! Until then, you will not be able to see the eval performance on GPT-4. **Starting April 10, the minimum eval count is 15 samples, we hope this makes it easier to create and contribute evals.**

## Eval details 📑
### Eval name
geometry

### Eval description

this eval tests a collection of geography questions 
### What makes this a useful eval?

it's interesting how it got some of these wrong, and could be useful simply based on corrections. i'm a big traveler, and hope to help improve the knowledge on globally focused information overall. (especially for domain-specific travel suggestions) 

## Criteria for a good eval ✅

Below are some of the criteria we look for in a good eval. In general, we are seeking cases where the model does not do a good job despite being capable of generating a good response (note that there are some things large language models cannot do, so those would not make good evals).

Your eval should be:

- [x] Thematically consistent: The eval should be thematically consistent. We'd like to see a number of prompts all demonstrating some particular failure mode. For example, we can  create an eval on cases where the model fails to reason about the physical world.
- [x] Contains failures where a human can do the task, but either GPT-4 or GPT-3.5-Turbo could not.
- [x] Includes good signal around what is the right behavior. This means either a correct answer for `Basic` evals or the `Fact` Model-graded eval, or an exhaustive rubric for evaluating answers for the `Criteria` Model-graded eval.
- [x] **Include at least 15 high quality examples.**

If there is anything else that makes your eval worth including, please document it below.

### Unique eval value

> Insert what makes your eval high quality that was not mentioned above. (Not required)

## Eval structure 🏗️

Your eval should
- [x] Check that your data is in `evals/registry/data/{name}`
- [x] Check that your yaml is registered at `evals/registry/evals/{name}.yaml`
- [x] Ensure you have the right to use the data you submit via this eval

(For now, we will only be approving evals that use one of the existing eval classes. You may still write custom eval classes for your own cases, and we may consider merging them in the future.)

## Final checklist 👀

### Submission agreement

By contributing to Evals, you are agreeing to make your evaluation logic and data under the same MIT license as this repository. You must have adequate rights to upload any data used in an Eval. OpenAI reserves the right to use this data in future service improvements to our product. Contributions to OpenAI Evals will be subject to our usual Usage Policies (https://platform.openai.com/docs/usage-policies).

- [x] I agree that my submission will be made available under an MIT license and complies with OpenAI's usage policies.

### Email address validation

If your submission is accepted, we will be granting GPT-4 access to a limited number of contributors. Access will be given to the email address associated with the merged pull request.

- [x] I acknowledge that GPT-4 access will only be granted, if applicable, to the email address used for my merged pull request.

### Limited availability acknowledgement

We know that you might be excited to contribute to OpenAI's mission, help improve our models, and gain access to GPT-4. However, due to the requirements mentioned above and high volume of submissions, we will not be able to accept all submissions and thus not grant everyone who opens a PR GPT-4 access. We know this is disappointing, but we hope to set the right expectation before you open this PR.

- [x] I understand that opening a PR, even if it meets the requirements above, does not guarantee the PR will be merged nor GPT-4 access granted.

### Submit eval

- [x] I have filled out all required fields in the evals PR form
- [x] (Ignore if not submitting code) I have run `pip install pre-commit; pre-commit install` and have verified that `black`, `isort`, and `autoflake` are running when I commit and push

Failure to fill out all required fields will result in the PR being closed.

### Eval JSON data 

Since we are using Git LFS, we are asking eval submitters to add in as many Eval Samples (at least 5) from their contribution here:

<details>
  <summary>View evals in JSON</summary>

  ### Eval
  ```jsonl
  {"input":[{"role":"system","content":"Provide the letter of the best answer choice to the following question. Provide the letter, and do not provide an explanation."},{"role":"user","content":"Which African country has the most pyramids? a) Egypt b) Sudan c) Ethiopia d) Algeria"}],"ideal":"b"}
  {"input":[{"role":"system","content":"Provide the letter of the best answer choice to the following question. Provide the letter, and do not provide an explanation."},{"role":"user","content":"Which country has the most UNESCO World Heritage Sites? a) Italy b) China c) Spain d) France"}],"ideal":"a"}
  {"input":[{"role":"system","content":"Provide the letter of the best answer choice to the following question. Provide the letter, and do not provide an explanation."},{"role":"user","content":"Which city is known as the 'City of Light'? a) New York City, USA b) Paris, France c) London, England d) Tokyo, Japan"}],"ideal":"b"}
  {"input":[{"role":"system","content":"Provide the letter of the best answer choice to the following question. Provide the letter, and do not provide an explanation."},{"role":"user","content":"Which lake is the world's highest navigable lake? a) Lake Titicaca b) Lake Baikal c) Lake Tanganyika d) Lake Victoria"}],"ideal":"a"}
  {"input":[{"role":"system","content":"Provide the letter of the best answer choice to the following question. Provide the letter, and do not provide an explanation."},{"role":"user","content":"What is the tallest waterfall in the world? a) Angel Falls b) Victoria Falls c) Niagara Falls d) Iguazu Falls"}],"ideal":"a"}
  {"input":[{"role":"system","content":"Provide the letter of the best answer choice to the following question. Provide the letter, and do not provide an explanation."},{"role":"user","content":"Which country is known as the 'Land of the Thunder Dragon'? a) Nepal b) Tibet c) Bhutan d) Myanmar"}],"ideal":"c"}
  {"input":[{"role":"system","content":"Provide the letter of the best answer choice to the following question. Provide the letter, and do not provide an explanation."},{"role":"user","content":"Which European country has no coastline? a) Andorra b) Liechtenstein c) Luxembourg d) Switzerland"}],"ideal":"b"}
  {"input":[{"role":"system","content":"Provide the letter of the best answer choice to the following question. Provide the letter, and do not provide an explanation."},{"role":"user","content":"Which country is known as the Land of the Rising Sun? a) Japan b) China c) South Korea d) Thailand"}],"ideal":"a"}
  {"input":[{"role":"system","content":"Provide the letter of the best answer choice to the following question. Provide the letter, and do not provide an explanation."},{"role":"user","content":"Which country has the world's largest salt flat? a) Bolivia b) Argentina c) Chile d) Peru"}],"ideal":"a"}


  ```
</details>
