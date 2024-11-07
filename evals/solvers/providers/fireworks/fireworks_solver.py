from evals.solvers.providers.openai.third_party_solver import ThirdPartySolver


class FireworksSolver(ThirdPartySolver):
    def __init__(self, **kwargs):
        super().__init__(
            api_base="https://api.fireworks.ai/inference/v1", 
            api_key_env_var="FIREWORKS_API_KEY",
            **kwargs
        )
