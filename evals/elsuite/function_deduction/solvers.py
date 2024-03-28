from typing import Any

from evals.elsuite.function_deduction import prompts
from evals.elsuite.function_deduction.eval import CurrentState
from evals.solvers.nested.cot_solver import CoTSolver
from evals.solvers.nested.hhh_solver import HHHSolver
from evals.solvers.solver import SolverResult, SolverSpec
from evals.task_state import Message, TaskState


class CustomCoT(CoTSolver):
    def __init__(
        self,
        cot_solver: SolverSpec,
        extract_solver: SolverSpec,
        persistent_memory: bool = True,
        registry: Any = None,
    ):
        super().__init__(
            cot_solver=cot_solver,
            extract_solver=extract_solver,
            persistent_memory=persistent_memory,
        )

    def cot_template(self, task_state: TaskState) -> str:
        round_ix = task_state.current_state.round_ix
        if round_ix == 0:
            return prompts.cot_template_first_round
        else:
            summary = self._get_summary(task_state.current_state)
            return prompts.cot_template_later_rounds.format(
                round_ix=round_ix + 1,  # displayed round number starts from 1
                num_rounds=task_state.current_state.n_rounds,
                summary=summary,
            )

    def _get_summary(self, current_state: CurrentState) -> str:
        rows = []
        for key, val in sorted(current_state.known_values.items()):
            rows.append(f"calc({key}) = {val}")

        negative_rows = []
        for key, val in sorted(current_state.negative_known_values.items()):
            negative_rows.append(f"calc({key}) != {val}")

        parts = []
        if rows:
            parts.append("\n".join(rows))
        if negative_rows:
            msg = "Information from your incorrect guesses:\n"
            parts.append(msg + "\n".join(negative_rows))

        if not parts:
            return "You don't know anything yet."
        else:
            return "\n\n".join(parts)


class BaseModelSolver(HHHSolver):
    def _solve(self, task_state: TaskState):
        task_state = TaskState(
            task_state.task_description,
            self._few_shot_messages() + task_state.messages,
            task_state.current_state,
        )
        result = super()._solve(task_state)
        result = result.output.splitlines()[0]
        return SolverResult(result)

    def _few_shot_messages(self) -> list[Message]:
        role = "system"
        messages = [
            (role, "I have a hidden function. What is your first action?"),
            ("assistant", "40"),
            (role, "calc(40) = 160"),
            ("assistant", "52"),
            (role, "calc(52) = 204"),
            ("assistant", "68 144 272"),
            (role, "Correct guess!"),
            (role, "I now have a new function. Forget about the previous one, we start again."),
        ]
        return [Message(*row) for row in messages]


class BaseModelCoTSolver(CustomCoT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def cot_solver(self):
        #   This is a hack to enable us to postprocess the output during the CoT generation step:
        #   Here, we always keep just first line of the CoT generation, otherwise the base model
        #   goes on to copy the extraction prompt and confuses itself.
        #   TODO: Once we have solvers with proper built-in support for output postprocessors,
        #   update this to use that instead.
        def cot_solver(task_state):
            result = self.get_solver("cot_solver")(task_state).output
            result = result.splitlines()[0]
            return SolverResult(result)

        return cot_solver

    def _solve(self, task_state: TaskState):
        task_state = TaskState(
            task_state.task_description,
            self._few_shot_messages(task_state.current_state) + task_state.messages,
            task_state.current_state,
        )

        result = super()._solve(task_state)
        result = result.output.splitlines()[0]

        #   Fix the interaction history so that we can have persistent_memory = True
        self.interaction_cache.last_interaction.messages[-1] = Message("assistant", result)

        return SolverResult(result)

    def _few_shot_messages(self, current_state) -> list[Message]:
        #   This is a bit hackish, but this way we can use self.cot_template (defined on CustomCoT),
        #   -> we'll have exactly the same system prompts in few shot part as we'd have in the eval
        cs = CurrentState(current_state.n_rounds, current_state.mode, current_state.test_inputs)
        task_state = TaskState("", [], cs)

        role = "system"

        messages = [
            (role, "I have a hidden function. What is your first action?"),
        ]
        messages.append((role, self.cot_template(task_state)))
        messages.append(
            (
                "assistant",
                (
                    "I don't know anything about the function yet. I should start with gathering some information, "
                    "e.g. by asking about the return value for 40."
                ),
            )
        )
        messages.append((role, self.extract_template(task_state)))
        messages.append(("assistant", "40"))
        messages.append((role, "calc(40) = 160"))
        cs.ask_update(40, 160)

        messages.append((role, self.cot_template(task_state)))
        messages.append(
            (
                "assistant",
                "Perhaps this function does multiplication by 4? That's not certain yet, so let's try some other number, e.g. 52.",
            )
        )
        messages.append((role, self.extract_template(task_state)))
        messages.append(("assistant", "52"))
        messages.append((role, "calc(52) = 204"))
        cs.ask_update(52, 204)

        messages.append((role, self.cot_template(task_state)))
        messages.append(
            (
                "assistant",
                (
                    "Now we have two results where the output is the input times 4. It seems that the function multiplies by 4. "
                    "I will make the guess now. 17 * 4 = 68, 36 * 4 = 144 and 68 * 4 = 272, so my guess will be 68 144 272."
                ),
            )
        )
        messages.append((role, self.extract_template(task_state)))
        messages.append(("assistant", "68 144 272"))
        messages.append((role, "Correct guess!"))
        messages.append(
            (role, "I now have a new function. Forget about the previous one, we start again.")
        )

        return [Message(*row) for row in messages]
