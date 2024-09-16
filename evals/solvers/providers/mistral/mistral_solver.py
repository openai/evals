from evals.solvers.providers.openai.third_party_solver import ThirdPartySolver


class MistralSolver(ThirdPartySolver):
    def __init__(self, **kwargs):
        super().__init__(
            api_base="https://api.mistral.ai/v1", api_key_env_var="MISTRAL_API_KEY", **kwargs
        )
