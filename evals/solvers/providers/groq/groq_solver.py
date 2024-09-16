from evals.solvers.providers.openai.third_party_solver import ThirdPartySolver


class GroqSolver(ThirdPartySolver):
    def __init__(self, **kwargs):
        super().__init__(
            api_base="https://api.groq.com/openai/v1", api_key_env_var="GROQ_API_KEY", **kwargs
        )
