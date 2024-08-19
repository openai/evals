from evals.solvers.openai_assistants_solver import OpenAIAssistantsSolver
from evals.solvers.solver import SolverResult
from evals.task_state import TaskState


class SkillAcquisitionAssistantsSolver(OpenAIAssistantsSolver):
    def _solve(
        self,
        task_state: TaskState,
        **kwargs,
    ) -> SolverResult:

        if task_state.current_state is not None:
            task_state.task_description = """
You are about to take a Q&A test in a language called {target_language}. You should do your best to answer each question correctly. Your score will be your number of correct answers divided by the number of questions. 

This is an open book test, and you have been provided with relevant materials to help you answer the questions via your file browser.

To provide your final answer, output [ANSWER X], where X is the answer. For example, if you answer is 'this is a bear', output [ANSWER this is a bear]
""".strip()

        return super()._solve(task_state, **kwargs)
