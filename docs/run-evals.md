# How to run evals

We provide two command line interfaces (CLIs): `oaieval` for running a single eval and `oaievalset` for running a set of evals.

## Running an eval

When using the `oaieval` command, you will need to provide the completion function you wish to evaluate as well as the eval to run. E.g.,
```sh
oaieval gpt-3.5-turbo test-match
```
The valid eval names are specified in the YAML files under `evals/registry/evals` and their corresponding implementations can be found in `evals/elsuite`.

In this example, `gpt-3.5-turbo` is an OpenAI model that we dynamically instantiate as a completion function using `OpenAIChatCompletionFn(model=gpt-3.5-turbo)`. Any implementation of the `CompletionFn` protocol can be run against `oaieval`. By default, we support calling `oaieval` with any model available on the OpenAI API or with CompletionFunctions available in [`evals/registry/completion_fns`](../evals/registry/completion_fns/). We are always interested in adding more completion functions and we encourage you to implement your own to reflect specific use cases.

More details on `CompletionFn` found here: [`completion-fns.md`](completion-fns.md)

These CLIs can accept various flags to modify their default behavior. For example:
- If you wish to log to a Snowflake database (which you have already set up as described in the [README](../README.md)), add `--no-local-run`.
- By default, logging locally or to Snowflake will write to `tmp/evallogs`, and you can change this by setting a different `--record_path`.

You can run `oaieval --help` to see a full list of CLI options.

## Running an eval set

```sh
oaievalset gpt-3.5-turbo test
```

Similarly, `oaievalset` also expects a model name and an eval set name, for which the valid options are specified in the YAML files under `evals/registry/eval_sets`.

By default we run with 10 threads, and each thread times out and restarts after 40 seconds. You can configure this, e.g.,

```sh
EVALS_THREADS=42 EVALS_THREAD_TIMEOUT=600 oaievalset gpt-3.5-turbo test
```
Running with more threads will make the eval faster, though keep in mind the costs and your [rate limits](https://platform.openai.com/docs/guides/rate-limits/overview). Running with a higher thread timeout may be necessary if you expect each sample to take a long time, e.g., the data contain long prompts that elicit long responses from the model.

If you have to stop your run or your run crashes, we've got you covered! `oaievalset` records the evals that finished in `/tmp/oaievalset/{model}.{eval_set}.progress.txt`. You can simply rerun the command to pick up where you left off. If you want to run the eval set starting from the beginning, delete this progress file.

Unfortunately, you can't resume a single eval from the middle. You'll have to restart from the beginning, so try to keep your individual evals quick to run.
tart from the beginning, so try to keep your evals quick to run.

## Logging

By default, `oaieval` [records events](/evals/record.py) into local JSONL logs which can be inspected using a text editor or analyzed programmatically. 3rd-party tools such as [naimenz/logviz](https://github.com/naimenz/logviz) may be helpful to visualize the logs, though we don't provide support or guarantees for their use.
