# OpenAI Evals

Evals is a framework for evaluating LLMs (large language models) or systems built using LLMs as components. It also includes an open-source registry of challenging evals.

We now support evaluating the behavior of any system including prompt chains or tool-using agents, via the [Completion Function Protocol](docs/completion-fns.md).

With Evals, we aim to make it as simple as possible to build an eval while writing as little code as possible. An "eval" is a task used to evaluate the quality of a system's behavior. To get started, we recommend that you follow these steps:

To get set up with evals, follow the [setup instructions below](README.md#Setup).

#### Running evals
- Learn how to run existing evals: [run-evals.md](docs/run-evals.md).
- Familiarize yourself with the existing eval templates: [eval-templates.md](docs/eval-templates.md).

#### Writing evals

**Important: Please note that we are currently not accepting Evals with custom code!** While we ask you to not submit such evals at the moment, you can still submit modelgraded evals with custom modelgraded YAML files.

- Walk through the process for building an eval: [build-eval.md](docs/build-eval.md)
- See an example of implementing custom eval logic: [custom-eval.md](docs/custom-eval.md).

#### Writing CompletionFns
- Write your own completion functions: [completion-fns.md](docs/completion-fns.md)

If you think you have an interesting eval, please open a PR with your contribution. OpenAI staff actively review these evals when considering improvements to upcoming models.

____________________
🚨 For a limited time, we will be granting GPT-4 access to those who contribute high quality evals. Please follow the instructions mentioned above and note that spam or low quality submissions will be ignored❗️

Access will be granted to the email address associated with an accepted Eval. Due to high volume, we are unable to grant access to any email other than the one used for the pull request.
____________________

## Setup

To run evals, you will need to set up and specify your OpenAI API key. You can generate one at <https://platform.openai.com/account/api-keys>. After you obtain an API key, specify it using the `OPENAI_API_KEY` environment variable. **Please be aware of the [costs](https://openai.com/pricing) associated with using the API when running evals.**

**Minimum Required Version: Python 3.9**

### Downloading evals

Our Evals registry is stored using [Git-LFS](https://git-lfs.com/). Once you have downloaded and installed LFS, you can fetch the evals (from within your local copy of the evals repo) with:
```sh
cd evals
git lfs fetch --all
git lfs pull
```

This will populate all the pointer files under `evals/registry/data`.

You may just want to fetch data for a select eval. You can achieve this via:
```sh
git lfs fetch --include=evals/registry/data/${your eval}
git lfs pull
```

### Making evals

If you are going to be creating evals, we suggest cloning this repo directly from GitHub and installing the requirements using the following command:

```sh
pip install -e .
```

Using `-e`, changes you make to your eval will be reflected immediately without having to reinstall.

Optionally, you can install the formatters for pre-committing with:

```sh
pip install -e .[formatters]
```

### Running evals

If you don't want to contribute new evals, but simply want to run them locally, you can install the evals package via pip:

```sh
pip install evals
```

We provide the option for you to log your eval results to a Snowflake database, if you have one or wish to set one up. For this option, you will further have to specify the `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_DATABASE`, `SNOWFLAKE_USERNAME`, and `SNOWFLAKE_PASSWORD` environment variables.

## FAQ

Do you have any examples of how to build an eval from start to finish?

- Yes! These are in the `examples` folder. We recommend that you also read through [build-eval.md](docs/build-eval.md) in order to gain a deeper understanding of what is happening in these examples.

Do you have any examples of evals implemented in multiple different ways?

- Yes! In particular, see `evals/registry/evals/coqa.yaml`. We have implemented small subsets of the [CoQA](https://stanfordnlp.github.io/coqa/) dataset for various eval templates to help illustrate the differences.

When I run an eval, it sometimes hangs at the very end (after the final report). What's going on?

- This is a known issue, but you should be able to interrupt it safely and the eval should finish immediately after.

There's a lot of code, and I just want to spin up a quick eval. Help? OR,

I am a world-class prompt engineer. I choose not to code. How can I contribute my wisdom?

- If you follow an existing [eval template](docs/eval-templates.md) to build a basic or model-graded eval, you don't need to write any evaluation code at all! Just provide your data in JSON format and specify your eval parameters in YAML. [build-eval.md](docs/build-eval.md) walks you through these steps, and you can supplement these instructions with the Jupyter notebooks in the `examples` folder to help you get started quickly. Keep in mind, though, that a good eval will inevitably require careful thought and rigorous experimentation!

## Disclaimer

By contributing to Evals, you are agreeing to make your evaluation logic and data under the same MIT license as this repository. You must have adequate rights to upload any data used in an Eval. OpenAI reserves the right to use this data in future service improvements to our product. Contributions to OpenAI Evals will be subject to our usual Usage Policies: https://platform.openai.com/docs/usage-policies.
