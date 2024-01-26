"""
This file defines various helper functions for interacting with the OpenAI API.
"""
import concurrent
import logging
import os
import time

import backoff
import openai
from openai import OpenAI

EVALS_THREAD_TIMEOUT = float(os.environ.get("EVALS_THREAD_TIMEOUT", "40"))


@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=(
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.InternalServerError,
    ),
    max_value=60,
    factor=1.5,
)
def openai_completion_create_retrying(client: OpenAI, *args, **kwargs):
    """
    Helper function for creating a completion.
    `args` and `kwargs` match what is accepted by `openai.Completion.create`.
    """
    result = client.completions.create(*args, **kwargs)
    if "error" in result:
        logging.warning(result)
        raise openai.error.APIError(result["error"])
    return result


def request_with_timeout(func, *args, timeout=EVALS_THREAD_TIMEOUT, **kwargs):
    """
    Worker thread for making a single request within allotted time.
    """
    while True:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                result = future.result(timeout=timeout)
                return result
            except concurrent.futures.TimeoutError:
                continue


@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=(
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.InternalServerError,
    ),
    max_value=60,
    factor=1.5,
)
def openai_chat_completion_create_retrying(client: OpenAI, *args, **kwargs):
    """
    Helper function for creating a chat completion.
    `args` and `kwargs` match what is accepted by `openai.ChatCompletion.create`.
    """
    result = request_with_timeout(client.chat.completions.create, *args, **kwargs)
    if "error" in result:
        logging.warning(result)
        raise openai.error.APIError(result["error"])
    return result


@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=(
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.InternalServerError,
    ),
    max_value=60,
    factor=1.5,
)
def openai_rag_completion_create_retrying(client: OpenAI, *args, **kwargs):
    """
    Helper function for creating a RAG completion.
    `args` and `kwargs` match what is accepted by `openai.ChatCompletion.create`.
    """

    file = client.files.create(file=open(kwargs["file_name"], "rb"), purpose='assistants')

    #  Create an Assistant (Note model="gpt-3.5-turbo-1106" instead of "gpt-4-1106-preview")
    assistant = client.beta.assistants.create(
        name="File Assistant",
        instructions=kwargs.get("instructions", ""),
        model=kwargs.get("model", "gpt-3.5-turbo-1106"),
        tools=[{"type": "retrieval"}],
        file_ids=[file.id]
    )

    #  Create a Thread
    thread = client.beta.threads.create()

    # Add a Message to a Thread
    message = client.beta.threads.messages.create(thread_id=thread.id, role="user",
                                                  content=kwargs.get("prompt", "")
                                                  )

    # Run the Assistant
    run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant.id)

    # If run is 'completed', get messages and print
    while True:
        # Retrieve the run status
        run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        time.sleep(10)
        if run_status.status == 'completed':
            messages = client.beta.threads.messages.list(thread_id=thread.id)
            answer = messages.data[0].content[0].text.value
            break
        else:
            ### sleep again
            time.sleep(2)

    # if "error" in result:
    #     logging.warning(result)
    #     raise openai.error.APIError(result["error"])
    return answer
