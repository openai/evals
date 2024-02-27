import json
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Union

from curl_cffi import requests as requests_cffi
from httpcore import HTTPProxy
from requests_toolbelt import MultipartEncoder

from evals.api import CompletionFn, CompletionResult
from evals.prompt.base import (
    ChatCompletionPrompt,
    CompletionPrompt,
    OpenAICreateChatPrompt,
    OpenAICreatePrompt,
    Prompt,
)
from evals.utils.doc_utils import extract_text_and_fill_in_images, extract_text
from evals.record import record_sampling

try:
    from claude2_api.client import (
        ClaudeAPIClient,
        SendMessageResponse,
    )
    from claude2_api.session import SessionData, get_session_data
    from claude2_api.errors import ClaudeAPIError, MessageRateLimitError, OverloadError
except ImportError:
    print("Run `pip install unofficial-claude2-api` to use Unofficial Claude API.")

try:
    import anthropic
except ImportError:
    print("Run `pip install anthropic` to use Official Claude API.")


class ClaudeAPIClientRenewed(ClaudeAPIClient):
    def __init__(self,
                 session: SessionData,
                 proxy: HTTPProxy | None = None,
                 model_name: str = "claude-2.1",
                 cache_dir: Optional[str] = str(Path.home() / ".claude/knowledge_base.json"),
                 timeout: float = 240):
        super().__init__(session, proxy, model_name, timeout)
        self.__session = self._ClaudeAPIClient__session
        self.__BASE_URL= self._ClaudeAPIClient__BASE_URL
        self.cache_dir = cache_dir
        Path(self.cache_dir).parent.mkdir(parents=True, exist_ok=True)
        if not Path(self.cache_dir).exists():
            json.dump({}, open(self.cache_dir, "w"))

    def __prepare_file_attachment(self, fpath: str, chat_id: str) -> dict | None:
        cache = json.load(open(self.cache_dir, 'r+'))

        if cache.get(fpath, None) is not None:
            pdf_info = cache[fpath]
            return pdf_info

        content_type = self._ClaudeAPIClient__get_content_type(fpath)
        if content_type == "text/plain":
            return self._ClaudeAPIClient__prepare_text_file_attachment(fpath)

        url = f"{self.__BASE_URL}/api/convert_document"

        headers = {
            "authority": "claude.ai",
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9,zh;q=0.8,zh-CN;q=0.7,ja;q=0.6,mt;q=0.5,it;q=0.4,da;q=0.3,zh-TW;q=0.2",
            "cookie": self.__session.cookie,
            "origin": self.__BASE_URL,
            "referer": f"{self.__BASE_URL}/chats",
            "sec-ch-ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "Connection": "keep-alive",
            "sec-fetch-site": "same-origin",
            "User-Agent": self.__session.user_agent,
            'Content-Type': content_type
        }

        # headers = {
        #     "Host": "claude.ai",
        #     "User-Agent": self.__session.user_agent,
        #     "Accept": "*/*",
        #     "Accept-Language": "en-US,en;q=0.5",
        #     "Accept-Encoding": "gzip, deflate, br",
        #     "Referer": f"{self.__BASE_URL}/chat/{chat_id}",
        #     "Origin": self.__BASE_URL,
        #     "DNT": "1",
        #     "Sec-Fetch-Dest": "empty",
        #     "Sec-Fetch-Mode": "cors",
        #     "Sec-Fetch-Site": "same-origin",
        #     "Connection": "keep-alive",
        #     "Cookie": self.__session.cookie,
        #     "TE": "trailers",
        # }

        with open(fpath, "rb") as fp:
            # files = {
            #     "file": (os.path.basename(fpath), fp, content_type),
            #     "orgUuid": (None, self.__session.organization_id),
            # }
            multipart_data = MultipartEncoder(
                fields={
                    "file": (os.path.basename(fpath), fp, content_type),
                    "orgUuid": self.__session.organization_id
                }
            )
            data = BytesIO(multipart_data.to_string())
            headers["Content-Type"] = multipart_data.content_type

            response = requests_cffi.post(
                url,
                data=data,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            if response.status_code == 200:
                cache[fpath] = response.json()
                json.dump(cache, open(self.cache_dir, "w"))
                return response.json()
        print(
            f"\n[{response.status_code}] Unable to prepare file attachment -> {fpath}\n"
            f"\nReason: {response.text}\n\n"
        )
        return None

    def send_message(
        self,
        chat_id: str,
        prompt: str,
        attachment_paths: list[str] = None,
    ) -> SendMessageResponse:
        """
        Send message to `chat_id` using specified `prompt` string.

        You can omitt or provide an attachments path list using `attachment_paths`

        Returns a `SendMessageResponse` instance, having:
        - `answer` string field,
        - `status_code` integer field,
        - `error_response` string field, which will be None in case of no errors.
        """

        self._ClaudeAPIClient__check_file_attachments_paths(attachment_paths)

        attachments = []
        if attachment_paths:
            attachments = [
                a
                for a in [
                    self.__prepare_file_attachment(path, chat_id)
                    for path in attachment_paths
                ]
                if a
            ]

        url = f"{self.__BASE_URL}/api/organizations/{self.__session.organization_id}/chat_conversations/{chat_id}/completion"

        payload = {
            "attachments": attachments,
            "files": [],
            "model": "claude-2.1",
            "prompt": prompt,
            "timezone": 'Asia/Shanghai',
        }

        headers = {
            "Host": "claude.ai",
            "User-Agent": self.__session.user_agent,
            "Accept": "text/event-stream, text/event-stream",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": f"{self.__BASE_URL}/chat/{chat_id}",
            "Origin": self.__BASE_URL,
            "DNT": "1",
            "Cookie": self.__session.cookie,
            "TE": "trailers",
        }

        response = requests_cffi.post(
            url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
            impersonate='chrome120'
        )
        response.raise_for_status()

        enc = None
        if "Content-Encoding" in response.headers:
            enc = response.headers["Content-Encoding"]

        # Decrypt encoded response
        dec = self._ClaudeAPIClient__decode_response(response.content, enc)

        return SendMessageResponse(
            self._ClaudeAPIClient__parse_send_message_response(dec),
            response.status_code,
            response.content,
        )


class ClaudeCompletionResult(CompletionResult):
    def __init__(self, response: str, prompt: Any) -> None:
        self.response = response
        self.prompt = prompt

    def get_completions(self) -> list[str]:
        return [self.response.strip()]


class ClaudeUnofficialCompletionFn(CompletionFn):
    def __init__(
        self,
        model: Optional[str] = "claude-2.1",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        n_ctx: Optional[int] = None,
        extra_options: Optional[dict] = {},
        **kwargs,
    ):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.n_ctx = n_ctx
        self.extra_options = extra_options

        claude_environs = {k: v for k, v in os.environ.items() if k.startswith("CLAUDE")}
        if len(claude_environs) != 3:
            self.session: SessionData = get_session_data()
        else:
            self.session: SessionData = SessionData(cookie=os.environ["CLAUDE_COOKIE"],
                                                    user_agent=os.environ["CLAUDE_USER_AGENT"],
                                                    organization_id=os.environ["CLAUDE_ORGID"])

    def __call__(
        self,
        prompt: Union[str, OpenAICreateChatPrompt],
        **kwargs,
    ) -> ClaudeCompletionResult:
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
        FILEPATH_LIST = [kwargs["file_name"]] if "file_name" in kwargs else None

        client = ClaudeAPIClientRenewed(self.session, timeout=240, model_name=self.model)
        chat_id = client.create_chat()
        if not chat_id:
            # This will not throw MessageRateLimitError
            # But it still means that account has no more messages left.
            print("\nMessage limit hit, cannot create chat...")
            result = ClaudeCompletionResult(response="Message limit hit, cannot create chat...", prompt=openai_create_prompt)
        else:
            try:
                # Used for sending message with or without attachments
                # Returns a SendMessageResponse instance
                res: SendMessageResponse = client.send_message(
                    chat_id, openai_create_prompt, attachment_paths=FILEPATH_LIST
                )
                # Inspect answer
                if res.answer:
                    result = ClaudeCompletionResult(response=res.answer, prompt=openai_create_prompt)
                else:
                    # Inspect response status code and raw answer bytes
                    print(f"\nError code {res.status_code}, raw_answer: {res.raw_answer}")
            except ClaudeAPIError as e:
                # Identify the error
                if isinstance(e, MessageRateLimitError):
                    # The exception will hold these informations about the rate limit:
                    print(f"\nMessage limit hit, resets at {e.reset_date}")
                    print(f"\n{e.sleep_sec} seconds left until -> {e.reset_timestamp}")
                elif isinstance(e, OverloadError):
                    print(f"\nOverloaded error: {e}")
                else:
                    print(f"\nGot unknown Claude error: {e}")
                result = ClaudeCompletionResult(response=str(e), prompt=openai_create_prompt)
            finally:
                # Perform chat deletion for cleanup
                client.delete_chat(chat_id)

        record_sampling(prompt=result.prompt, sampled=result.get_completions())
        return result


class ClaudeOfficialCompletionFn(CompletionFn):
    def __init__(
        self,
        model: Optional[str] = "claude-2.1",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        n_ctx: Optional[int] = None,
        extra_options: Optional[dict] = {},
        **kwargs,
    ):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key or os.environ.get("CLAUDE_API_KEY", None)
        self.n_ctx = n_ctx
        self.extra_options = extra_options

    def __call__(
        self,
        prompt: Union[str, OpenAICreateChatPrompt],
        **kwargs,
    ) -> ClaudeCompletionResult:
        if not isinstance(prompt, Prompt):
            assert (
                isinstance(prompt, str)
                or (isinstance(prompt, list) and all(isinstance(token, int) for token in prompt))
                or (isinstance(prompt, list) and all(isinstance(token, str) for token in prompt))
                or (isinstance(prompt, list) and all(isinstance(msg, dict) for msg in prompt))
            ), f"Got type {type(prompt)}, with val {type(prompt[0])} for prompt, expected str or list[int] or list[str] or list[dict[str, str]]"

        if "file_name" in kwargs:
            attached_file_content = "\nThe file is as follows:\n\n" + "".join(extract_text(kwargs["file_name"]))
        else:
            attached_file_content = ""

        client = anthropic.Anthropic(api_key=self.api_key)

        if isinstance(prompt, str):
            prompt = CompletionPrompt(
                raw_prompt=prompt,
            )

            openai_create_prompt: OpenAICreatePrompt = prompt.to_formatted_prompt() + attached_file_content

            response = client.completions.create(max_tokens_to_sample=1024, model=self.model, prompt=openai_create_prompt)
        else:
            prompt = ChatCompletionPrompt(raw_prompt=prompt)
            openai_create_prompt: OpenAICreateChatPrompt = prompt.to_formatted_prompt()
            openai_create_prompt[-1]["content"] += attached_file_content

            response = client.messages.create(max_tokens=1024, model=self.model, messages=openai_create_prompt)

        result = ClaudeCompletionResult(response=response.text, prompt=prompt)

        record_sampling(prompt=result.prompt, sampled=result.get_completions())
        return result
