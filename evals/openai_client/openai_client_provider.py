import os
from abc import ABC

from openai import OpenAI
from openai.lib.azure import AzureOpenAI


class OpenAIClientProvider(ABC):
    def __init__(self, api_key, **kwargs):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.azure_endpoint = kwargs.pop("azure_endpoint", None) or os.environ.get(
            "OPENAI_AZURE_ENDPOINT"
        )
        self.api_version = kwargs.pop("opena_api_version", None) or os.environ.get(
            "OPENAI_API_VERSION"
        )

        self.kwargs = kwargs

    def get_client(self):
        if not self.azure_endpoint:
            return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        return AzureOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            azure_endpoint=os.environ.get("OPENAI_AZURE_ENDPOINT"),
            api_version=os.environ.get("OPENAI_API_VERSION"),
        )
