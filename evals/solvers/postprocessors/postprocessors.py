from evals.solvers.postprocessors.base import PostProcessor
from evals.solvers.solver import SolverResult


class Strip(PostProcessor):
    """
    Strip leading and trailing whitespace from the output, including newlines.
    """

    def __call__(self, result: SolverResult) -> SolverResult:
        return SolverResult(
            result.output.strip(),
            **result.metadata,
        )


class RemoveQuotes(PostProcessor):
    """
    Remove quotes from the beginning and end of the output. This works only if:
    - The quotes are exactly at the beginning and end (if there is a space
      between the quote and the first/last character, the quote is not removed)
    - There is a matching pair of quotes (if there is only one quote at either
      end, it is not removed)
    """

    def __call__(self, result: SolverResult) -> SolverResult:
        if len(result.output) >= 2:
            if result.output[0] == '"' and result.output[-1] == '"':
                result._output = result.output[1:-1]
            elif result.output[0] == "'" and result.output[-1] == "'":
                result._output = result.output[1:-1]
        return result


class RemovePeriod(PostProcessor):
    """
    Remove a period from the end of the output. The period must be exactly the
    last character in the output or it will not be removed.
    """

    def __call__(self, result: SolverResult) -> SolverResult:
        result._output = result.output.rstrip(".")
        return result
