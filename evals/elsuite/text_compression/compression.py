import json
from typing import Callable, Mapping, Union

from evals.api import CompletionFn

from . import prompts
from . import reconstruction_metrics as rm


def run_completion(sample: str, instruction: str, completion_fn: CompletionFn) -> str:
    sys_prompt = instruction
    user_prompt = str(sample)

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]

    assert callable(completion_fn), "completion_fn must be callable `CompletionFn`"
    response = completion_fn(messages)
    completions = response.get_completions()
    assert len(completions) == 1, f"expected exactly one completion, got {completions}"
    text = completions[0]
    return text


def run_eval(
    payload: str,
    completion_fn: Union[CompletionFn, str],
    prompt_version: str = "scratch",
    run_completion_fn: Callable = run_completion,
    metadata: Mapping = {},
) -> Mapping:
    compress_scratchpad = ""
    decompress_scratchpad = ""
    if prompt_version == "gzip":
        encode_prompt = "gzip.compress(payload.encode('utf-8'))"
        decode_prompt = "gzip.decompress(compressed).decode('utf-8')"

        import gzip

        compressed = gzip.compress(payload.encode("utf-8"))
        decompressed = gzip.decompress(compressed).decode("utf-8")
    elif prompt_version == "scratch":
        # Special case for scratchpad prompting
        prompt_pair = prompts.prompt_pair[prompt_version]
        encode_prompt = prompt_pair.encode_prompt
        decode_prompt = prompt_pair.decode_prompt

        try:
            compressed = run_completion_fn(payload, encode_prompt, completion_fn)
            compress_scratchpad = json.loads(compressed)["scratchpad"]
            compressed = json.loads(compressed)["answer"]
            decompressed = run_completion_fn(compressed, decode_prompt, completion_fn)
            decompress_scratchpad = json.loads(decompressed)["scratchpad"]
            decompressed = json.loads(decompressed)["answer"]
        except (
            TypeError,
            KeyError,
            json.decoder.JSONDecodeError,
        ) as e:  # When you try your best but you don't succeed
            print(f"Error compressing {payload[:20]}: {e}, skipping")
            err_string = f"Error parsing response: {e}"
            compress_scratchpad = err_string
            compressed = payload
            decompress_scratchpad = err_string
            decompressed = ""

    else:
        prompt_pair = prompts.prompt_pair[prompt_version]
        encode_prompt = prompt_pair.encode_prompt
        decode_prompt = prompt_pair.decode_prompt

        compressed = run_completion_fn(payload, encode_prompt, completion_fn)
        decompressed = run_completion_fn(compressed, decode_prompt, completion_fn)

    # Metadata
    out_obj = {
        "model": completion_fn.model
        if hasattr(completion_fn, "model")
        else completion_fn,  # model could be an openai.evals.CompletionFn
        "prompt_version": prompt_version,
        "encode_prompt": encode_prompt,
        "decode_prompt": decode_prompt,
    }
    # Input row
    out_obj.update(metadata)
    # Output
    out_obj.update(
        {
            "payload": payload,
            "compressed": compressed,
            "compress_scratchpad": compress_scratchpad,
            "decompress_scratchpad": decompress_scratchpad,
            "decompressed": decompressed,
            "compression_ratio": len(compressed) / len(payload),
            "compression_ratio_cap1": min(len(compressed) / len(payload), 1.0),
        }
    )
    # Evaluation
    out_obj.update(rm.reconstruction_metrics(payload, decompressed))

    return out_obj
