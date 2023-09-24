# Building an eval

**Important: Please note that we are currently not accepting Evals with custom code!** While we ask you to not submit such evals at the moment, you can still submit modelgraded evals with custom modelgraded YAML files.

This document walks through the end-to-end process for building an eval, which is a dataset and a choice of eval class. The `examples` folder contains Jupyter notebooks that follow the steps below to build several academic evals, thus helping to illustrate the overall process.

The steps in this process are building your dataset, registering a new eval with your dataset, and running your eval. Crucially, we assume that you are using an [existing eval template](eval-templates.md) out of the box (if that's not the case, see [this example of building a custom eval](custom-eval.md)). If you are interested in contributing your eval publicly, we also include some criteria at the bottom for what we think makes an interesting eval.

We are looking for evals in the following categories:

- Over-refusals
- Safety
- System message steerability
- In-the-wild hallucinations
- Math / logical / physical reasoning
- Real-world use case (please describe in your PR how this capability would be used in a product)
- Other foundational capability

If you have an eval that falls outside this category but still is a diverse example, please contribute it!

## Formatting your data

Once you have an eval in mind that you wish to implement, you will need to convert your samples into the right JSON lines (JSONL) format. A JSONL file is just a JSON file with a unique JSON object per line.

You can use the `openai` CLI (available with [OpenAI-Python](https://github.com/openai/openai-python)) to transform data from some common file types into JSONL:
``` 
openai tools fine_tunes.prepare_data -f data[.csv, .json, .txt, .xlsx or .tsv]
```

We include some examples of JSONL eval files in [registry/data/README.md](../evals/registry/data/README.md)

Each JSON object will represent one data point in your eval. The keys you need in the JSON object depend on the eval template. All templates expect an `"input"` key, which is the prompt, ideally specified in [chat format](https://platform.openai.com/docs/guides/chat/introduction) (though strings are also supported). We recommend chat format even if you are evaluating non-chat models. If you are evaluating both chat and non-chat models, we handle the conversion between chat-formatted prompts and raw string prompts (see the conversion logic [here](../evals/prompt/base.py)).

For the basic evals `Match`, `Includes`, and `FuzzyMatch`, the other required key is `"ideal"`, which is a string (or a list of strings) specifying the correct reference answer(s). For model-graded evals, the required keys vary based on the eval but is determined by the `{key}`s in the evaluation `prompt` that are not covered by the (optional) `args`.

We have implemented small subsets of the [CoQA](https://stanfordnlp.github.io/coqa/) dataset for various eval templates to illustrate how the data should be formatted. See [`coqa/match.jsonl`](../evals/registry/data/coqa/match.jsonl) for an example of data that is suitable for the `Match` basic eval template and [`coqa/samples.jsonl`](../evals/registry/data/coqa/samples.jsonl) for data that is suitable for `fact` and `closedqa` model-graded evals. Note that even though these two model-graded evals expect different keys, we can include the superset of keys in our data in order to support both evals.

If the dataset file is on your local machine, put the `jsonl` file in `evals/registry/data/<eval_name>/samples.jsonl`. If it is in Cloud Object Storage, we support path-style URLs for the major clouds (for your personal use only, we will not accept PRs with cloud URLs).

## Registering the eval

Register the eval by adding a file to `evals/registry/evals/<eval_name>.yaml` using the elsuite registry format. For example, for a `Match` eval, it would be:
```
<eval_name>:
  id: <eval_name>.dev.v0
  description: <description>
  metrics: [accuracy]

<eval_name>.dev.v0:
  class: evals.elsuite.basic.match:Match
  args:
    samples_jsonl: <eval_name>/samples.jsonl
```

Upon running the eval, the data will be searched for in `evals/registry/data`. For example, if `test_match/samples.jsonl` is the provided filepath, the data is expected to be in `evals/registry/data/test_match/samples.jsonl`.

The naming convention for evals is in the form `<eval_name>.<split>.<version>`.
- `<eval_name>` is the eval name, used to group evals whose scores are comparable.
- `<split>` is the data split, used to further group evals that are under the same `<base_eval>`. E.g., "val", "test", or "dev" for testing.
- `<version>` is the version of the eval, which can be any descriptive text you'd like to use (though it's best if it does not contain `.`).

In general, running the same eval name against the same model should always give similar results so that others can reproduce it. Therefore, when you change your eval, you should bump the version.

## Running the eval

You can now run your eval on your data from the CLI with your choice of model or completion function:
```
oaieval gpt-3.5-turbo <eval_name>
```
Congratulations, you have built your eval! Keep iterating on it until you are confident in the results.

## For model-graded evals: a step-by-step workflow

We expect that the existing model-graded evals such as `fact`, `closedqa`, and `battle` will fit many use cases. However, other use cases may benefit from more customization, e.g., a different evaluation prompt. For these, there will be a bit more work involved, but generally still no coding required!

1. If you can't use an existing model-graded eval, create a new YAML or create a new entry to an existing YAML in `evals/registry/modelgraded` to specify the [parameters](eval-templates.md#parameters-for-model-graded-evals) of your eval. See [`humor.yaml`](../evals/registry/modelgraded/humor.yaml) for an example.
    - Note that, even if you are creating a new YAML, you may find it easiest to copy an existing YAML as a starting point. For example, model-graded evals which check a model completion against a rubric can copy `closedqa.yaml` and just edit the `args`.
2. Next, you will create your dataset and register your eval, as described above. See [`joke_fruits_labeled.jsonl`](../evals/registry/data/test_metaeval/joke_fruits_labeled.jsonl) and [`joke-fruits`](../evals/registry/evals/test-modelgraded.yaml), for example.
    - Note that it is recommended to specify `eval_type` at this step, when you register your eval, rather than step 1.
3. Run your eval, e.g., `oaieval gpt-3.5-turbo joke-fruits`.
4. (Recommended) Add a meta-eval for the model-graded eval! Each model-graded eval comes with a few knobs to tune, mainly `prompt` but also `eval_type`. In order to make sure the eval is of high quality, we recommend each model-graded eval contribution come with "choice labels", which are basically human-provided labels for which evaluation choice the model should have made. As an example (pretending that these jokes are actually funny), see the `"choice"` keys in [`joke_fruits_labeled.jsonl`](../evals/registry/data/test_metaeval/joke_fruits_labeled.jsonl), which are not used by the `joke-fruits` eval but are used by the [`joke-fruits-meta`](../evals/registry/evals/test-modelgraded.yaml) meta-eval right below it . After running the meta-eval, e.g., `oaieval gpt-3.5-turbo joke-fruits-meta`, the report will output `metascore/` accuracies, which should be close to "1.0" for a good model-graded eval.

## Criteria for contributing an eval

Important: if you are contributing code, make sure to run `pip install pre-commit; pre-commit install` before committing and pushing to ensure that `black`, `isort`, and `autoflake` are run.

We are interested in curating a diverse and interesting set of evals on which to improve our models going forward. Here are some criteria for what we consider a good eval:
- [ ] The eval should be thematically consistent. We'd like to see a number of prompts all revolving around the same use case, subject domain, failure mode, etc.
- [ ] The eval should be challenging. If GPT-4 or GPT-3.5-Turbo do well on all of the prompts, this is not as interesting. Of course, the eval should also be possible given the models' limitations and constraints. Oftentimes, a good rule of thumb is whether a human (potentially a subject expert) could do well on the prompts.
- [ ] The eval should be directionally clear. The data should include good signal around what is the right behavior. This means, for example, high-quality reference answers or an exhaustive rubric for evaluating answers.
- [ ] The eval should be carefully crafted. Before you submit, you should think through whether you have engineered your prompts for good performance, whether you are using the best eval template, whether you have spot checked your results to ensure accuracy, etc.

Once you are ready to contribute your eval publicly, submit a PR and the OpenAI team will be happy to look it over. Make sure to fill out all parts of the template that is prepopulated into the PR message. Note that submitting a PR does not guarantee that OpenAI will eventually merge it. We will run our own checks and use our best judgment when considering which evals to follow up with.
