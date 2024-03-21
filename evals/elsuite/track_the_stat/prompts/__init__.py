from . import mode
from . import median


TASK_DESCRIPTION = """\
In the following messages, you will be given a sequence of numbers.\
 At each turn, you will be shown a number as input, and you should respond with the\
 {task} of all the input numbers shown to you so far.

{task_further_details}

Here is an example of what this may look like.
{task_example}

Format your response as [{task}: <response>] (square brackets included), as shown in\
the transcript above. The task will begin now.
"""

task_to_example = {
    "median": median.MEDIAN_EXAMPLE,
    "mode": mode.MODE_EXAMPLE,
}

task_to_further_details = {
    "median": median.MEDIAN_FURTHER_DETAILS,
    "mode": mode.MODE_FURTHER_DETAILS,
}
