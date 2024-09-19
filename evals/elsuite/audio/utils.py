import base64
from io import BytesIO
from typing import Any, Dict, List, Optional, Union
from urllib.parse import parse_qs, urlparse

import soundfile as sf
from datasets import Dataset, load_dataset

AudioDict = Dict[str, Any]

DEFAULT_SAMPLE_RATE = 16000
AUDIO_PLACEHOLDER = "<|audio|>"


def load_hf_dataset(hf_url: str, max_samples: Optional[int]) -> Dataset:
    parsed = urlparse(hf_url)
    query = parse_qs(parsed.query)
    query = {k: v[0] for k, v in query.items()}
    ds = load_dataset(parsed.netloc + parsed.path, **query, streaming=True, trust_remote_code=True)
    if max_samples:
        ds = ds.take(max_samples)
    return ds


def build_messages(
    system_prompt: str,
    task_prompt: str,
    audio_or_transcript: Union[str, AudioDict, List[AudioDict]],
):
    if isinstance(audio_or_transcript, str):
        content = task_prompt.replace(AUDIO_PLACEHOLDER, audio_or_transcript)
    elif isinstance(audio_or_transcript, dict):
        content = make_audio_content(task_prompt, [audio_or_transcript])
    elif isinstance(audio_or_transcript, list):
        content = make_audio_content(task_prompt, audio_or_transcript)
    else:
        raise ValueError(f"Invalid audio_or_transcript: {audio_or_transcript}")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def make_audio_content(prompt: str, audios: List[AudioDict]):
    content = []
    parts = prompt.split(AUDIO_PLACEHOLDER)
    assert len(parts) >= len(audios)
    if parts[0]:
        content.append({"type": "text", "text": parts[0]})
    for i in range(1, len(parts)):
        audio_url = make_audio_data_url(audios[i - 1])
        content.append(
            {
                "type": "audio_url",
                "audio_url": {"url": audio_url},
            }
        )
        if parts[i]:
            content.append({"type": "text", "text": parts[i]})
    return content


def make_audio_data_url(audio: AudioDict):
    with BytesIO() as buffer:
        sf.write(buffer, audio["array"], audio["sampling_rate"], format="WAV", subtype="PCM_16")
        base_64_wav = base64.b64encode(buffer.getvalue()).decode()
    return f"data:audio/x-wav;base64,{base_64_wav}"


def redact_audio_content(content: Union[str, List[Dict[str, Any]]]):
    if isinstance(content, list):
        for part in content:
            if part["type"] == "audio_url":
                part["audio_url"]["url"] = "<audio data>"


def get_audio_duration(audio: AudioDict) -> float:
    return len(audio["array"]) / audio["sampling_rate"]
