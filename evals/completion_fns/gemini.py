import os
import numpy as np
from typing import Any, Optional, Union

from evals.api import CompletionFn, CompletionResult
from evals.prompt.base import (
    CompletionPrompt,
    OpenAICreateChatPrompt,
    OpenAICreatePrompt,
    Prompt,
)
from evals.utils.api_utils import request_with_timeout
from evals.utils.doc_utils import extract_text_and_fill_in_images, extract_text
from evals.record import record_sampling

try:
    import google.generativeai as genai
except ImportError:
    print("Run `pip install google-generativeai` to use Gemini API.")


class GeminiCompletionResult(CompletionResult):
    def __init__(self, response: str, prompt: Any) -> None:
        self.response = response
        self.prompt = prompt

    def get_completions(self) -> list[str]:
        return [self.response.strip()]


class GeminiCompletionFn(CompletionFn):
    def __init__(
        self,
        model: Optional[str] = "gemini-pro",
        api_base: Optional[str] = None,
        api_keys: Optional[list[str], str] = None,
        n_ctx: Optional[int] = None,
        extra_options: Optional[dict] = {},
        **kwargs,
    ):
        self.model = model
        self.api_base = api_base
        self.api_keys = [api_keys] if type(api_keys) == str else [api_keys] or os.environ.get("GOOGLE_API_KEY").split(",")
        self.n_ctx = n_ctx
        self.extra_options = extra_options

    def __call__(
        self,
        prompt: Union[str, OpenAICreateChatPrompt],
        **kwargs,
    ) -> GeminiCompletionResult:
        if not isinstance(prompt, Prompt):
            assert (
                isinstance(prompt, str)
                or (isinstance(prompt, list) and all(isinstance(token, int) for token in prompt))
                or (isinstance(prompt, list) and all(isinstance(token, str) for token in prompt))
                or (isinstance(prompt, list) and all(isinstance(msg, dict) for msg in prompt))
            ), f"Got type {type(prompt)}, with val {type(prompt[0])} for prompt, expected str or list[int] or list[str] or list[dict[str, str]]"

            prompt = CompletionPrompt(
                raw_prompt=prompt,
            )

        openai_create_prompt: OpenAICreatePrompt = prompt.to_formatted_prompt()

        if "file_name" in kwargs:
            attached_file_content = ["The file is as follows:"]
            attached_file_content += extract_text_and_fill_in_images(kwargs["file_name"], None, False) \
                if self.model == "gemini-pro-vision" else ["".join(extract_text(kwargs["file_name"]))]
        else:
            attached_file_content = []
            self.model = "gemini-pro"

        genai.configure(api_key=np.random.choice(self.api_keys))

        generation_config = {
            "temperature": 0.4,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 8192,
        }
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
        ]
        model = genai.GenerativeModel(model_name=self.model,
                                      generation_config=generation_config,
                                      safety_settings=safety_settings)
        response = request_with_timeout(model.generate_content, contents=[openai_create_prompt] + attached_file_content)
        # response = model.generate_content(
        #     contents=[openai_create_prompt] + attached_file_content
        # )
        answer = response.text

        record_sampling(prompt=prompt, sampled=answer)
        return GeminiCompletionResult(response=answer, prompt=prompt)
