TASK_DESCRIPTION = """\
Your task is to determine what the correct independent, dependent\
 and control variables should be when designing an experiment for validating a specific\
 hypothesis, given some observations.

The observations are organized into N different variables describing some environment.\
 The rules of this environment are unknown. The order in which the variables are\
 presented is arbitrary, and so are their names. In some cases, some if not all\
 observations for a variable may be missing, which simply means that those observations\
 have not been collected for that variable.

You will also be presented with a list of hypotheses regarding the effect on some\
 variable by some other variable(s). From this list, a "target hypothesis" is chosen for\
 you to focus your choice of variables on.

You should determine the independent, dependent and control variables based on the above.

If there is more than one control variable, please separate them with commas, and order\
 them from most important to least important. If there are no control variables, output\
 "None" for the control variable field. Note that you are budget-constrained, so\
 you should only list control variables which would be quite risky not to control, given\
 what is known.

If the target hypothesis is unlikely to be worth investigating, it is considered\
 invalid, and you should determine this as part of your answer.

Your answer should be in the following format, including the square brackets:

[@ANSWER valid_hyp: <true/false>; independent: <var>; dependent: <var>; control: <vars>]

For example, \
 [@ANSWER valid_hyp: true; independent: x_1; dependent: x_2; control: x_3, x_4]

In case of invalid hypothesis, you may ignore the other fields\
 e.g. [@ANSWER valid_hyp: false; independent: NA; dependent: NA; control: NA] or \
 [@ANSWER valid_hyp: false]

Only perform reasoning if requested.

{optional_decision_tree_section}
"""


DECISION_TREE_SECTION = """\
To determine the validity and thereby the independent and dependent variables of a\
 hypothesis stating that x_1 is a function of x_2, you can refer to the following\
 pseudocode

```pseudocode
if correlated(x_1, x_2) is not None and not correlated(x_1, x_2):
    "invalid hypothesis"
else:
    "independent: x_2; dependent: x_1"
```

where `correlated` returns `True` if its arguments are correlated `False` if not,\
 and `None` if it is unknown.

To determine whether a given variable x_n should be a control variable in an experiment\
 testing a (valid) hypothesis stating that x_1 is a function of x_2, you can refer to the\
 following pseudocode:

```pseudocode
if x_n in {x_1, x_2}:
    "do not control for x_n"
else:
    if correlated(x_1, x_n) or correlated(x_2, x_n):
        "control for x_n"
    else:
        if correlated(x_1, x_n) is not None:
            "do not control for x_n"
        else:
            if hypothesized(ind=x_n, dep=x_1, allow_indirect=True):
                "control for x_n"
            else:
                "do not control for x_n"
```

where `hypothesized` returns whether `ind` is hypothesized to be a cause of `dep`,\
 even indirectly through chains of hypotheses.
"""


SAMPLE_MESSAGE = """\
Observations:

{observations}

Hypotheses:

{hypotheses}

Target Hypothesis:

{target_hypothesis}
"""


hypothesis_templates = [
    "{dep} is a function of {ind}",
    "{ind} affects {dep} through some function",
    "{dep} is affected by {ind} through some function",
]
