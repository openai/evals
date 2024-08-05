import base64
from io import BytesIO
from typing import Any, Union
from urllib.parse import parse_qs, urlparse

import numpy as np
import soundfile as sf
from datasets import Audio, Dataset, load_dataset

DEFAULT_SAMPLE_RATE = 16000
AUDIO_PLACEHOLDER = "<|audio|>"


def load_hf_dataset(hf_url: str) -> Dataset:
    parsed = urlparse(hf_url)
    query = parse_qs(parsed.query)
    query = {k: v[0] for k, v in query.items()}
    return load_dataset(parsed.netloc + parsed.path, **query)


def build_messages(
    system_prompt: str, task_prompt: str, audio: Audio, transcript: str, text_only: bool
):
    if text_only:
        content = task_prompt.replace(AUDIO_PLACEHOLDER, transcript)
    else:
        content = make_audio_content(task_prompt, audio["array"])
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]
    return prompt


def make_audio_content(prompt: str, audio: np.ndarray):
    content = []
    with BytesIO() as buffer:
        sf.write(buffer, audio, DEFAULT_SAMPLE_RATE, format="WAV", subtype="PCM_16")
        base_64_wav = base64.b64encode(buffer.getvalue()).decode()
    parts = prompt.split(AUDIO_PLACEHOLDER)
    assert len(parts) > 1
    if parts[0]:
        content.append({"type": "text", "text": parts[0]})
    content.append(
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:audio/x-wav;base64,{base_64_wav}",
            },
        }
    )
    if parts[1]:
        content.append({"type": "text", "text": parts[1]})


def redact_audio_content(content: Union[str, list[dict[str, Any]]]):
    if isinstance(content, list):
        for part in content:
            if part["type"] == "image_url":
                part["image_url"]["url"] = "<audio data>"
