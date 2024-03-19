# Postprocessors

Postprocessors are an output-tidying step for solvers. Many solvers, especially ones based on generative language models, generate answers that may be correct in essence but are not in the expected format. Postprocessors are useful for applying common string-processing operations to clean up the output for easy evaluation.

For example, a multiple-choice answer evaluation may require the solver to answer with `A`, `B` or `C` but a language model solver may output an answer like `"B."`. An exact match criterion may lead to a false negative even if `B` is the correct answer; a postprocessor is helpful in this case to remove the `""` quotes and `.` period to make it match the expected format.

## Usage

Postprocessors can be applied by passing a list of `path:Class` strings via the `postprocessors` argument of any Solver class, i.e. via the Solver YAML arguments. 

For example, in [`defaults.yaml`](/evals/registry/solvers/defaults.yaml) we have:
```yaml
generation/direct/gpt-3.5-turbo:
  class: evals.solvers.openai_solver:OpenAISolver
  args:
    completion_fn_options:
      model: gpt-3.5-turbo-0125
      extra_options:
        temperature: 1
        max_tokens: 512
    postprocessors: &postprocessors
      - evals.solvers.postprocessors.postprocessors:Strip
      - evals.solvers.postprocessors.postprocessors:RemoveQuotes
      - evals.solvers.postprocessors.postprocessors:RemovePeriod
```

**Note: The order of operations in applying postprocessors matters.** Postprocessors are applied in the order they are listed. In the above example, `Strip` is applied first, followed by `RemoveQuotes` and then `RemovePeriod`. This sequence works well for common cases such as when the answer has the form: `\n"<answer>."\n`.

## Available Postprocessors

Please see [`evals/solvers/postprocessors/postprocessors.py`](/evals/registry/solvers/postprocessors/postprocessors.py) for currently implemented postprocessors. You can also add your own postprocessors by subclassing `PostProcessor` in [`evals/solvers/postprocessors/base.py`](/evals/registry/solvers/postprocessors/base.py) and implementing the `__call__` method.
