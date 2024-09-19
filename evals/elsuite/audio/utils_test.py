import numpy as np

from evals.elsuite.audio.utils import (
    build_messages,
    get_audio_duration,
    make_audio_content,
    make_audio_data_url,
    redact_audio_content,
)


def test_build_messages():
    system_prompt = "You are a helpful assistant."
    task_prompt = "Transcribe this audio: <|audio|>"

    # Test with string
    messages = build_messages(system_prompt, task_prompt, "audio transcript")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Transcribe this audio: audio transcript"

    # Test with Audio object
    audio = {"array": np.zeros(1600), "sampling_rate": 16000}
    messages = build_messages(system_prompt, task_prompt, audio)
    assert len(messages) == 2
    assert isinstance(messages[1]["content"], list)

    # Test with list of Audio objects
    task_prompt = "Start <|audio|> Middle <|audio|> End"
    messages = build_messages(system_prompt, task_prompt, [audio, audio])
    assert len(messages) == 2
    assert isinstance(messages[1]["content"], list)
    assert len(messages[1]["content"]) == 5  # text, audio, text, audio, text


def test_make_audio_content_audio_only():
    prompt = "<|audio|>"
    audio = {"array": np.zeros(1600), "sampling_rate": 16000}
    content = make_audio_content(prompt, [audio])
    assert len(content) == 1
    assert content[0]["type"] == "audio_url"


def test_make_audio_content_text_then_audio():
    prompt = "Start <|audio|>"
    audio = {"array": np.zeros(1600), "sampling_rate": 16000}
    content = make_audio_content(prompt, [audio])
    assert len(content) == 2
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "audio_url"


def test_make_audio_content_text_and_audio():
    prompt = "Start <|audio|> Middle <|audio|> End"
    audio = {"array": np.zeros(1600), "sampling_rate": 16000}
    content = make_audio_content(prompt, [audio, audio])
    assert len(content) == 5
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "audio_url"
    assert content[2]["type"] == "text"
    assert content[3]["type"] == "audio_url"
    assert content[4]["type"] == "text"


def test_make_audio_data_url():
    audio = {"array": np.zeros(1600), "sampling_rate": 16000}
    url = make_audio_data_url(audio)
    assert url.startswith("data:audio/x-wav;base64,")


def test_redact_audio_content():
    content = [
        {"type": "text", "text": "Start"},
        {"type": "audio_url", "audio_url": {"url": "some_url"}},
        {"type": "text", "text": "End"},
    ]
    redact_audio_content(content)
    assert content[1]["audio_url"]["url"] == "<audio data>"


def test_get_audio_duration():
    audio = {"array": np.zeros(1600), "sampling_rate": 16000}
    duration = get_audio_duration(audio)
    assert duration == 0.1
