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


def truncate_multimodal_prompt(prompt_list, max_images=16, max_size_bytes=4 * 1024 * 1024):
    """
    Truncates a list of texts and images to meet the constraints of maximum images and total size in bytes.

    Parameters:
    - prompt_list: List containing texts and images. Images are expected to be dictionaries with keys 'mime_type' and 'data'.
    - max_images: Maximum number of images allowed.
    - max_size_bytes: Maximum total size allowed in bytes.

    Returns:
    - A truncated list that fits the constraints.
    """
    truncated_list = []
    total_size = 0
    image_count = 0

    for item in prompt_list:
        if isinstance(item, str):  # It's text
            item_size = len(item.encode('utf-8'))  # Size in bytes
        elif isinstance(item, dict) and item.get('mime_type') and item.get('data'):  # It's an image
            # The image data is a string representation of bytes; calculate its length accordingly.
            item_size = len(item['data'])  # Approximation of size in bytes
            image_count += 1
        else:
            continue  # Skip any item that doesn't fit expected structure

        # Check if adding this item would exceed limits
        if total_size + item_size > max_size_bytes or image_count > max_images:
            break  # Stop adding items

        total_size += item_size
        truncated_list.append(item)

    return truncated_list


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

            contents = [openai_create_prompt] + attached_file_content

            if self.model == "gemini-pro":
                while num_tokens_from_string(contents[2], "cl100k_base") > max_tokens:
                    contents[2] = contents[2][:-1000]
            elif self.model == "gemini-pro-vision":
                contents = truncate_multimodal_prompt(contents, max_images=16, max_size_bytes=4 * 1024 * 1024)
        else:
            contents = [openai_create_prompt]
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
            contents=contents,
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
