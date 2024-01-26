"""
Extending Completion Functions with Embeddings-based retrieval from a fetched dataset
"""
import json
import os
import time
from pathlib import Path

import requests
from typing import Any, Optional, Union

from evals.prompt.base import CompletionPrompt
from evals.api import CompletionFn, CompletionResult
from evals.record import record_sampling


class UniFinderCompletionResult(CompletionResult):
    def __init__(self, response: str) -> None:
        self.response = response

    def get_completions(self) -> list[str]:
        return [self.response.strip()] if self.response else ["Unknown"]


class UniFinderCompletionFn(CompletionFn):
    """
    This Completion Function uses embeddings to retrieve the top k relevant docs from a dataset to the prompt, then adds them to the context before calling the completion.
    """

    def __init__(
            self,
            model: Optional[str] = None,
            instructions: Optional[str] = "You are a helpful assistant on extracting information from files.",
            api_base: Optional[str] = None,
            api_key: Optional[str] = None,
            n_ctx: Optional[int] = None,
            cache_dir: Optional[str] = str(Path.home() / ".uni_finder/knowledge_base.json"),
            pdf_parse_mode: Optional[str] = 'fast',  # or 'precise', 指定使用的pdf解析版本
            extra_options: Optional[dict] = {},
            **kwargs
    ):
        self.model = model
        self.instructions = instructions
        self.api_base = api_base or os.environ.get("UNIFINDER_API_BASE")
        self.api_key = api_key or os.environ.get("UNIFINDER_API_KEY")
        self.n_ctx = n_ctx
        self.extra_options = extra_options
        self.cache_dir = cache_dir
        self.pdf_parse_mode = pdf_parse_mode
        Path(self.cache_dir).parent.mkdir(parents=True, exist_ok=True)
        if not Path(self.cache_dir).exists():
            json.dump({}, open(self.cache_dir, "w"))

    def __call__(self, prompt: Union[str, list[dict]], **kwargs: Any) -> UniFinderCompletionResult:
        """
        Args:
            prompt: The prompt to complete, in either text string or Chat format.
            kwargs: Additional arguments to pass to the completion function call method.
        """

        pdf_token = []
        if "file_name" in kwargs:
            cache = json.load(open(self.cache_dir, 'r+'))

            if cache.get(kwargs["file_name"], {}).get(self.pdf_parse_mode, None) is None:
                url = f"{self.api_base}/api/external/upload_pdf"
                files = {'file': open(kwargs["file_name"], 'rb')}
                data = {
                    'pdf_parse_mode': self.pdf_parse_mode,
                    'api_key': self.api_key
                }
                response = requests.post(url, data=data, files=files)
                pdf_id = response.json()['pdf_token']  # 获得pdf的id，表示上传成功，后续可以使用这个id来指定pdf

                if kwargs["file_name"] not in cache:
                    cache[kwargs["file_name"]] = {self.pdf_parse_mode: pdf_id}
                else:
                    cache[kwargs["file_name"]][self.pdf_parse_mode] = pdf_id
                json.dump(cache, open(self.cache_dir, "w"))
            else:
                pdf_id = cache[kwargs["file_name"]][self.pdf_parse_mode]
            print("############# pdf_id ##############", pdf_id)
            pdf_token.append(pdf_id)

        url = f"{self.api_base}/api/external/chatpdf"

        if type(prompt) == list:
            prompt = CompletionPrompt(prompt).to_formatted_prompt()

        payload = {
            "model_engine": self.model,
            "pdf_token": pdf_token,
            "query": prompt,
            'api_key': self.api_key
        }
        response = requests.post(url, json=payload, timeout=1200)
        try:
            answer = response.json()['answer']
        except:
            print(response.text)
            answer = response.text
        record_sampling(prompt=prompt, sampled=answer)
        return UniFinderCompletionResult(answer)
