from typing import Optional

from evals.solvers.providers.openai.third_party_solver import ThirdPartySolver


class FixieSolver(ThirdPartySolver):
    def __init__(self, api_base: Optional[str] = None, **kwargs):
        super().__init__(
            api_base or "https://ultravox.api.fixie.ai/v1", "ULTRAVOX_API_KEY", **kwargs
        )
