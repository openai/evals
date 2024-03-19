from typing import List


def list_to_nl_list(list_of_words: List[str]) -> str:
    """
    Converts a list of words into a natural language list.
    """
    if len(list_of_words) == 1:
        return list_of_words[0]
    elif len(list_of_words) == 2:
        return f"{list_of_words[0]} and {list_of_words[1]}"
    else:
        return f"{', '.join(list_of_words[:-1])} and {list_of_words[-1]}"


OPENING_STATEMENT = """\
The following is a description of the observations made about a set of variables.
""".strip()

MANY_CORREL_SETS_MAIN = """\
In general, there were cases where some variables changed in tandem with each other,\
 while others did not.
""".strip()

SINGLE_CORREL_SET_MAIN = """\
In general, all of the variables seemed to change in tandem with each other.
For example, {example_1} {optional_example_2} {optional_concluding_statement}
""".strip()

ONLY_IND_MAIN = """\
In general, no discernible patterns were noticed between the variables.
For example, {example_1} {optional_example_2} {optional_concluding_statement}
""".strip()

CORREL_VARS_EXAMPLE = """\
{optional_transition}changes in {var_1} were observed to reflect changes in {var_2} and\
 viceversa.
""".strip()

IND_VARS_EXAMPLE = """\
{optional_transition}changes in {var_1} were not observed to reflect any changes in\
 {var_2}.
""".strip()

SPECIFIC_CONCL_STATEMENT = """\
Similar observations were made for all other pairings within and across\
 {already_mentioned} and {remaining_vars}.
""".strip()

GENERIC_CONCL_STATEMENT = """\
Similar observations were made for all other pairings of the observed variables.
""".strip()

UNOBS_BUT_HYP_VARS = """\
{unobs_but_hyp_vars} {be_string} not observed but {be_string} hypothesized about.
""".strip()
