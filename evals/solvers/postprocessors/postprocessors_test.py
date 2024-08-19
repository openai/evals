from evals.solvers.postprocessors.postprocessors import RemovePeriod, RemoveQuotes, Strip
from evals.solvers.solver import SolverResult


def test_strip():
    result = SolverResult("  abc  ")
    assert Strip()(result).output == "abc"
    result = SolverResult("abc")
    assert Strip()(result).output == "abc"
    result = SolverResult("")
    assert Strip()(result).output == ""
    result = SolverResult(" ")
    assert Strip()(result).output == ""


def test_remove_quotes():
    result = SolverResult('"abc"')
    assert RemoveQuotes()(result).output == "abc"
    result = SolverResult("'abc'")
    assert RemoveQuotes()(result).output == "abc"
    result = SolverResult("abc")
    assert RemoveQuotes()(result).output == "abc"
    result = SolverResult("abc'")
    assert RemoveQuotes()(result).output == "abc'"
    result = SolverResult("abc'abc'abc")
    assert RemoveQuotes()(result).output == "abc'abc'abc"
    result = SolverResult("")
    assert RemoveQuotes()(result).output == ""
    result = SolverResult("''")
    assert RemoveQuotes()(result).output == ""
    result = SolverResult("'" + "something" + '"')
    assert RemoveQuotes()(result).output == "'" + "something" + '"'


def test_remove_period():
    result = SolverResult("abc.")
    assert RemovePeriod()(result).output == "abc"
    result = SolverResult("abc")
    assert RemovePeriod()(result).output == "abc"
    result = SolverResult("abc.abc")
    assert RemovePeriod()(result).output == "abc.abc"
    result = SolverResult("")
    assert RemovePeriod()(result).output == ""
    result = SolverResult(".")
    assert RemovePeriod()(result).output == ""
    result = SolverResult(".5")
    assert RemovePeriod()(result).output == ".5"


def test_combination():
    sequence = [Strip(), RemoveQuotes(), RemovePeriod()]

    result = SolverResult("  'abc.'  ")
    for proc in sequence:
        result = proc(result)
    assert result.output == "abc"

    result = SolverResult("abc.''  ")
    for proc in sequence:
        result = proc(result)
    assert result.output == "abc.''"

    result = SolverResult("  ''.abc.'  ")
    for proc in sequence:
        result = proc(result)
    assert result.output == "'.abc"
