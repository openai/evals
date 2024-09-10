import base64
from io import BytesIO
from typing import Any, Dict, Optional, Union
from urllib.parse import parse_qs, urlparse

import numpy as np
import soundfile as sf
from datasets import Audio, Dataset, load_dataset

DEFAULT_SAMPLE_RATE = 16000
AUDIO_PLACEHOLDER = "<|audio|>"


def load_hf_dataset(hf_url: str, max_samples: Optional[int]) -> Dataset:
    parsed = urlparse(hf_url)
    query = parse_qs(parsed.query)
    query = {k: v[0] for k, v in query.items()}
    ds = load_dataset(parsed.netloc + parsed.path, **query, streaming=True)
    if max_samples:
        ds = ds.take(max_samples)
    return ds


def build_messages(system_prompt: str, task_prompt: str, audio_or_transcript: Union[str, Audio]):
    if isinstance(audio_or_transcript, str):
        content = task_prompt.replace(AUDIO_PLACEHOLDER, audio_or_transcript)
    else:
        content = make_audio_content(task_prompt, audio_or_transcript["array"])
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
            "type": "audio_url",
            "audio_url": {
                "url": f"data:audio/x-wav;base64,{base_64_wav}",
            },
        }
    )
    if parts[1]:
        content.append({"type": "text", "text": parts[1]})
    return content


def redact_audio_content(content: Union[str, list[dict[str, Any]]]):
    if isinstance(content, list):
        for part in content:
            if part["type"] == "audio_url":
                part["audio_url"]["url"] = "<audio data>"


def get_audio_duration(audio: Dict[str, Union[int, np.ndarray]]) -> float:
    return len(audio["array"]) / audio["sampling_rate"]
