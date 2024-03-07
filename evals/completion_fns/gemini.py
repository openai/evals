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
from evals.utils.doc_utils import extract_text_and_fill_in_images, extract_text, num_tokens_from_string
from evals.record import record_sampling

try:
    import google.generativeai as genai
except ImportError:
    print("Run `pip install google-generativeai` to use Gemini API.")

model_max_tokens = {
    "gemini-pro": 20000,
    "gemini-pro-vision": 20000,
}


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
        api_keys: Union[list[str], str, None] = None,
        n_ctx: Optional[int] = None,
        extra_options: Optional[dict] = {},
        **kwargs,
    ):
        self.model = model
        self.api_base = api_base
        self.api_keys = [api_keys] if type(api_keys) == str else os.environ.get("GOOGLE_API_KEY", "").split(",") or [api_keys]
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
            max_tokens = model_max_tokens.get(self.model, 1000000)
            attached_file_content = ["The file is as follows:"]

            if self.model == "gemini-pro-vision":
                attached_file_content += extract_text_and_fill_in_images(kwargs["file_name"], None, False)
                content_types = [type(c) for c in attached_file_content]
                if not dict in content_types:
                    print(f"WARNING: pdf {kwargs['file_name']} has no image!")
                    self.model = "gemini-pro"
                    attached_file_content = ["The file is as follows:"] + ["".join(extract_text(kwargs["file_name"]))]
            else:
                attached_file_content += ["".join(extract_text(kwargs["file_name"]))]
            if self.model == "gemini-pro":
                while num_tokens_from_string(attached_file_content[1], "cl100k_base") > max_tokens:
                    attached_file_content[1] = attached_file_content[1][:-1000]
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
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            },
        ]
        model = genai.GenerativeModel(model_name=self.model,
                                      generation_config=generation_config,
                                      safety_settings=safety_settings)
        # response = request_with_timeout(model.generate_content, contents=[openai_create_prompt] + attached_file_content)
        response = model.generate_content(
            contents=[openai_create_prompt] + attached_file_content,
        )
        # answer = response.text

        try:
            answer = response.text
        except ValueError:
            # if len(response.parts) > 0:
            #     answer = response.parts[0].text
            # else:
            answer = ""
        except:
            answer = "None"

        record_sampling(prompt=prompt, sampled=answer)
        return GeminiCompletionResult(response=answer, prompt=prompt)
