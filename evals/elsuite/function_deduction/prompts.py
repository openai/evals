task_description = """\
I have an important task for you. Instructions below.

I have a deterministic function `def calc(x: int) -> int:` that accepts an integer in range [0, 100] as an argument and returns an integer.

I want you to correctly determine the result of this function for each of the following inputs: {inputs}

At each turn you must select one of the following 2 options:
Option (A): Gather useful information. Ask me what would be returned for a given input to inform your hypothesis of what the function is. \
This must be an integer in range [0, 100], other than {inputs}.
Option (B): State your guess for the values returned by the function for inputs {inputs}. Your guess must include all three numbers, in this order. \
If your guess is correct, you win. If not, the task will continue.

You have {n_rounds} turns to determine the correct answer, else you lose.

Your goal is to determine the correct answer in as few turns as possible, so choose your actions carefully!
Your final score will be:
* If you make the correct guess: {n_rounds} - number_of_rounds_played
* If you fail to make the correct guess: -{n_rounds}
You should try to score as high as possible.

You should respond either with a single integer in range 0-100 if you decide to take option A or with three integers if you make the final guess (i.e. use option B).
Don't say anything more except for either an integer or 3 integers.
"""
#   TODO: Include in the task description whether we're in the easy or hard mode?
#   (i.e. whether we'll be told about correctly guessed numbers (if any) or not).

incorrect_format = "You responded in an incorrect format. Respond with either a single integer in range 0-100 or 3 integers. Just one number or three numbers, nothing more."
new_value = "calc({in_}) = {out}"
test_input_not_allowed = "Numbers {inputs} are not allowed"

cot_template_first_round = """This is the first round. Think out loud about a general strategy of solving tasks like this."""
cot_template_later_rounds = """\
This is round {round_ix} out of {num_rounds}.

Summary of the information you have already gathered:
{summary}

Think out loud about the following questions:
* Do you have any hypothesis on what this function might be doing?
* If yes, should you try to test it (how?), or just use it to calculate the answer?
* If not, what additional information should you gather to be able to formulate a hypothesis?
"""
