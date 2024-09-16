from evals.solvers.providers.openai.third_party_solver import ThirdPartySolver


class NvidiaSolver(ThirdPartySolver):
    def __init__(self, **kwargs):
        super().__init__(
            api_base="https://integrate.api.nvidia.com/v1",
            api_key_env_var="NVIDIA_API_KEY",
            **kwargs
        )
