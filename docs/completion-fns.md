# Completion Functions

## What are completion functions
In [run-evals.md](run-evals.md), we learned how to make calls to `oaieval` to run an eval against a completion function. Completion Functions are generalizations of model completions, where a "completion" is some text output that would be our answer to the prompt. For example, if "Who played the girl elf in the hobbit?" is our prompt, the correct completion is "Evangeline Lilly". While we can just test a model directly to see if it generates "Evangeline Lilly", we can imagine doing numerous other operations under the hood to improve our ability to answer this question, like giving the model access to a browser to look up the answer before responding. Making it easy to implement this kind of under-the-hood operators before responding is the motivation behind building Completion Functions.

## How to implement completion functions
A completion function needs to implement some interfaces that make it usable within Evals. At its core, it is just standardizing inputs to be a text string or [Chat conversation](https://platform.openai.com/docs/guides/chat), and the output to be a list of text strings. Implementing this interface will allow you to run your Completion Function against any eval in Evals.

The exact interfaces needed are described in detail in [completion-fn-protocol.md](completion-fn-protocol.md)

We include some example implementations inside `evals/completion_fns`. For example, the [`LangChainLLMCompletionFn`](../evals/completion_fns/langchain_llm.py) implements a way to generate completions from [LangChain LLMs](https://python.langchain.com/en/latest/modules/models/llms/getting_started.html). We can then use these completion functions with `oaieval`:
```
oaieval langchain/llm/flan-t5-xl test-match
```

## Registering Completion Functions
Once you have written a completion function, we need to make the class visible to the `oaieval` CLI. Similar to how we register our evals, we also register Completion Functions inside `evals/registry/completion_fns` as `yaml` files. Here is the registration for our langchain LLM completion function:
```yaml
langchain/llm/flan-t5-xl:
  class: evals.completion_fns.langchain_llm:LangChainLLMCompletionFn
  args:
    llm: HuggingFaceHub
    llm_kwargs:
      repo_id: google/flan-t5-xl
```
Here is how it breaks down
`langchain/llm/flan-t5-xl`: This is the top level key that will be used to access this completion function with `oaieval`.
`class`: This is the path to your implementation of the completion function protocol. This class needs to be importable within your python environment.
`args`:  These are arguments that are passed to your completion function when it is instantiated.


### Developing Completion Functions outside of Evals
It is possible to register CompletionFunctions without directly modifying the registry or code inside `Evals` by using the `--registry_path` argument. As an example, let's say I want to use `MyCompletionFn` located inside `~/my_project/`:
```
my_project
├── my_completion_fn.py
└── completion_fns
    └── my_completion_fn.yaml
```

If `my_project` is importable within the python environment (accessible via PYTHONPATH), we can structure `my_completion_fn.yaml` as:
```
my_completion_fn:
  class: my_project.my_completion_fn:MyCompletionFn
```
Then, we can make calls to `oaieval` using:
```
oaieval my_completion_fn test-match --registry_path ~/my_project
```
