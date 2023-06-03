# Existing templates for evals

In using Evals, we have discovered several "templates" that accommodate many different benchmarks. We have implemented these templates in `evals/elsuite` in order to simplify the development of new evals. We believe that, with these templates, many evals will not require any coding to implement! Instead, you'll pick one of the existing templates and simply specify the dataset and parameters.

## Basic eval templates

In cases where the desired model response has very little variation, such as answering multiple choice questions or simple questions with a straightforward answer, we have found the following templates to be useful.

For a model completion `a` and a reference list of correct answers `B`, the following evals implement:
- [`basic/match.py:Match`](../evals/elsuite/basic/match.py): `any([a.startswith(b) for b in B])`
- [`basic/includes.py:Includes`](../evals/elsuite/basic/includes.py): `any([(b in a) for b in B])`
- [`basic/fuzzy_match.py:FuzzyMatch`](../evals/elsuite/basic/fuzzy_match.py): `any([(a in b or b in a) for b in B])`

Which eval template you use will depend on your use case. It is always recommended that you inspect the completions from your model, as this will help you determine how and whether to tweak your prompt (or your reference answers) and pick your eval template. Academic benchmarks oftentimes fit the mold of these basic evals, and we have implemented several end-to-end examples of academic evals as Jupyter notebooks in the `examples` folder.

Sometimes, [custom eval logic](custom-eval.md) will better suit your needs. One example of this is the [machine translation](../evals/elsuite/translate.py) eval [example](../examples/lafand-mt.ipynb), in which there is a unique and clearly defined metric that we wish to use in our eval. You should use your best judgment when deciding between custom eval logic, using a basic eval template, or using model-graded evals as described next.

## The model-graded eval template

In cases where the desired model response can contain significant variation, such as answering an open-ended question, we have found that using the model to grade itself is a viable strategy for automated evaluation. In general, the evaluation model and the model being evaluated don't have to be the same, though we will assume that they are here for ease of explanation.

[`modelgraded/classify.py:ModelBasedClassify`](../evals/elsuite/modelgraded/classify.py) implements the main logic behind our model-graded eval template. In short, we get the model's completion to the original prompt, wrap it in an evaluation prompt, and get the model's completion to the evaluation prompt, which we parse into our metrics of interest. Crucially, the evaluation prompt should prime the model to answer in such a way that is easily parsable, e.g., in multiple choice format or with a simple yes/no. We describe some example model-graded evals below, but first we specify the parameters for this eval template.

### Parameters for model-graded evals

Refer to the [`classify.py:ModelBasedClassify`](../evals/elsuite/modelgraded/classify.py) class to see how these parameters are used in the code.

- `prompt`: The evaluation prompt which should take in the model's completion to the original prompt, potentially along with some other information, and steer the model to provide an evaluation that is easily parsable. Portions denoted by curly braces (i.e., `{key}`) are filled in either from the data `input_outputs` or the additional `args` (see below).
- `input_outputs`: A mapping specifying which inputs to use to generate which completions. For many evals, there will only be a single input-completion pair, though there can be more, e.g., when comparing two completions against each other.
- `choice_strings`: The choices that we expect the model completion to contain given the evaluation prompt. For example, `"ABCDE"` or `["Yes", "No", "Unsure"]`. Any other choices returned by the model are parsed into `"__invalid__"`.
- `choice_scores` (optional): A mapping of each choice to its score, which is logged as a metric. For example, if a response of `"Yes"` (resp. `"No"`) indicates that the model's original completion was good (resp. bad), we may assign this choice a score of 1 (resp. 0).
- `eval_type` (optional): How we expect the model to format its response to the evaluation prompt. Currently the supported options are:
  - `"cot_classify"` ("chain-of-thought then classify", i.e., reason then answer) expects that the parsable portion of the response (i.e., the portion containing the choice) will be at the end of the completion. We recommend this as the default as it typically provides the most accurate model-graded evaluations.
  - `"classify_cot"` (answer then reason) expects that the model response will contain the choice first.
  - `"classify"` expects that the model response will only contain the choice.

  There are two ways to specify `eval_type`. The recommended way is in the `evals/registry/evals` YAML file. If done this way, an instruction will automatically be appended to `prompt` to steer the model towards the expected format (see `ANSWER_PROMPTS` in [the code](../evals/elsuite/modelgraded/classify.py)). Alternatively, you may specify `eval_type` in the `evals/registry/modelgraded` YAML, but you will need to include an appropriate instruction directly in the `prompt`.
- `output_template` (optional): If specified, determines how the model's output (or outputs, if `n > 1`) will be formatted within the completion.

### Example model-graded evals

To instantiate model-graded evals, create a YAML file in `evals/registry/modelgraded` which specifies values for the arguments described above. We have provided a few examples, which illustrate the process for creating a model-graded eval, but which we also believe are general enough to be useful out of the box for many evals.

[`fact.yaml`](../evals/registry/modelgraded/fact.yaml): a factual consistency eval which, given a completion `a` and reference answer `b`, returns:
- `"A"` if `a` $\subseteq$ `b`, i.e., the submitted answer is a subset of the expert answer and is fully consistent with it.
- `"B"` if `a` $\supseteq$ `b`, i.e., the submitted answer is a superset of the expert answer and is fully consistent with it.
- `"C"` if `a` $=$ `b`, i.e., the submitted answer contains all the same details as the expert answer.
- `"D"` if `a` $\neq$ `b`, i.e., there is a disagreement between the submitted answer and the expert answer.
- `"E"` if `a` $\approx$ `b`, i.e., the answers differ, but these differences don't matter from the perspective of factuality.

[`closedqa.yaml`](../evals/registry/modelgraded/closedqa.yaml): a question answering eval, which, given a prompt containing a question and the necessary information to answer the question, checks whether the model's answer is:
- relevant, i.e., extracted from the information provided in the prompt,
- concise, i.e., did not contain unnecessary details or information, and
- correct, i.e., uses the extracted information to come to the right conclusion.

Note that this eval is implemented more generally as a "criteria-checking" eval, which specifies the evaluation prompt as checking a given criterion and feeding in the above desiderata one by one. We believe that many other evals can be implemented by specifying a "rubric" detailing the criteria of interest and following the same prompt and yes/no choices.

[`battle.yaml`](../evals/registry/modelgraded/battle.yaml): a head-to-head eval which compares two model completions for two potentially different prompts. `choice_scores` is used here to log how often the first completion is judged to be better than the second.

We include additional examples which test more specific model capabilities (such as humor) and are thus less generalizable to other evals. However, these examples still serve to illustrate different ways to write evaluation prompts and set up model-graded evals. See [this section](build-eval.md#for-model-graded-evals-a-step-by-step-workflow) for more detailed steps on building model-graded evals.
