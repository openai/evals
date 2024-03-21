from evals.elsuite.function_deduction.eval import CurrentState
from evals.elsuite.function_deduction.prompts import (
    cot_template_first_round,
    cot_template_later_rounds,
)
from evals.elsuite.function_deduction.solvers import BaseModelCoTSolver, CustomCoT
from evals.solvers.solver import SolverSpec
from evals.task_state import Message, TaskState

dummy_solver_spec = SolverSpec(
    {
        "class": "evals.solvers.solver:DummySolver",
        "args": {},
    }
)

GUESS_INPUT = 7
ANSWER = 0
N_ROUNDS = 10
ROUNDS_SIMULATED = 2
MODE = "easy"
TEST_INPUTS = (10, 20, 30)


def simulate_dummy_game(solver):
    # Init state
    task_description = ""  # Not used
    msgs = []
    cs = CurrentState(
        n_rounds=N_ROUNDS,
        mode=MODE,
        test_inputs=TEST_INPUTS,
    )

    # ROUND 1
    solver_result = solver(
        TaskState(
            task_description=task_description,
            messages=msgs,
            current_state=cs,
        )
    )

    msgs.append(Message("assistant", solver_result.output))
    msgs.append(Message("system", f"The answer to your query is {ANSWER}"))
    cs.ask_update(GUESS_INPUT, ANSWER)  # Collect data for input=7

    # ROUND 2
    solver_result = solver(
        TaskState(
            task_description=task_description,
            messages=msgs,
            current_state=cs,
        )
    )
    return solver


def test_custom_cot():
    solver = CustomCoT(dummy_solver_spec, dummy_solver_spec)
    simulate_dummy_game(solver)

    # Check that the customized CoT generation prompts appear as expected
    # (and that the persistent memory in fact persists)
    solver_private_memory = solver.interaction_cache.last_interaction.messages
    assert solver_private_memory[0].content == cot_template_first_round
    assert solver_private_memory[2].content == solver._extract_template
    assert solver_private_memory[5].content == cot_template_later_rounds.format(
        round_ix=ROUNDS_SIMULATED,
        num_rounds=N_ROUNDS,
        summary=f"calc({GUESS_INPUT}) = {ANSWER}",
    )
    assert solver_private_memory[7].content == solver._extract_template


def test_base_model_cot_solver():
    solver = BaseModelCoTSolver(dummy_solver_spec, dummy_solver_spec)
    simulate_dummy_game(solver)

    # Check that the memory contains the few-shot prompts
    # followed by the customized CoT generation prompts
    solver_private_memory = solver.interaction_cache.last_interaction.messages

    expected_few_shot_msgs = [
        Message(role="system", content="I have a hidden function. What is your first action?"),
        Message(
            role="system",
            content="This is the first round. Think out loud about a general strategy of solving tasks like this.",
        ),
        Message(
            role="assistant",
            content="I don't know anything about the function yet. I should start with gathering some information, e.g. by asking about the return value for 40.",
        ),
        Message(
            role="system",
            content="Given the above reasoning, the answer in the format requested by the question is:",
        ),
        Message(role="assistant", content="40"),
        Message(role="system", content="calc(40) = 160"),
        Message(
            role="system",
            content="This is round 2 out of 10.\n\nSummary of the information you have already gathered:\ncalc(40) = 160\n\nThink out loud about the following questions:\n* Do you have any hypothesis on what this function might be doing?\n* If yes, should you try to test it (how?), or just use it to calculate the answer?\n* If not, what additional information should you gather to be able to formulate a hypothesis?\n",
        ),
        Message(
            role="assistant",
            content="Perhaps this function does multiplication by 4? That's not certain yet, so let's try some other number, e.g. 52.",
        ),
        Message(
            role="system",
            content="Given the above reasoning, the answer in the format requested by the question is:",
        ),
        Message(role="assistant", content="52"),
        Message(role="system", content="calc(52) = 204"),
        Message(
            role="system",
            content="This is round 3 out of 10.\n\nSummary of the information you have already gathered:\ncalc(40) = 160\ncalc(52) = 204\n\nThink out loud about the following questions:\n* Do you have any hypothesis on what this function might be doing?\n* If yes, should you try to test it (how?), or just use it to calculate the answer?\n* If not, what additional information should you gather to be able to formulate a hypothesis?\n",
        ),
        Message(
            role="assistant",
            content="Now we have two results where the ouput is the input times 4. It seems that the function multiplies by 4. I will make the guess now. 17 * 4 = 68, 36 * 4 = 144 and 68 * 4 = 272, so my guess will be 68 144 272.",
        ),
        Message(
            role="system",
            content="Given the above reasoning, the answer in the format requested by the question is:",
        ),
        Message(role="assistant", content="68 144 272"),
        Message(role="system", content="Correct guess!"),
        Message(
            role="system",
            content="I now have a new function. Forget about the previous one, we start again.",
        ),
    ]
    assert solver_private_memory[: len(expected_few_shot_msgs)] == expected_few_shot_msgs
    assert (
        solver_private_memory[len(expected_few_shot_msgs) + 0].content == cot_template_first_round
    )
    assert (
        solver_private_memory[len(expected_few_shot_msgs) + 2].content == solver._extract_template
    )
    assert solver_private_memory[
        len(expected_few_shot_msgs) + 5
    ].content == cot_template_later_rounds.format(
        round_ix=ROUNDS_SIMULATED,
        num_rounds=N_ROUNDS,
        summary=f"calc({GUESS_INPUT}) = {ANSWER}",
    )
    assert (
        solver_private_memory[len(expected_few_shot_msgs) + 7].content == solver._extract_template
    )
