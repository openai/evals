import json
import logging
from typing import Any, Dict, List, Union
from urllib.parse import parse_qs, urlparse

import numpy as np
from datasets import load_dataset
from pydantic import BaseModel

import evals
import evals.metrics
from evals.api import CompletionFn
from evals.record import RecorderBase

logger = logging.getLogger(__name__)
DEFAULT_SAMPLE_RATE = 16000


class Sample(BaseModel):
    messages: List[dict[str, Any]]
    functions: List[dict]


class ToolsTask(evals.Eval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        dataset: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        *args,
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert len(completion_fns) == 1, "Audio tasks only support one completion fn"
        self.dataset = dataset
        self.temperature = temperature
        self.max_tokens = max_tokens

    def load_sample(self, row) -> list[Sample]:
        messages = row["messages"]
        last_tool_call_id = None
        for m in messages:
            if m["tool_calls"]:
                last_tool_call_id = m["tool_calls"][-1]["id"]
            elif m.get("tool_call_id") is not None and last_tool_call_id is not None:
                m["tool_call_id"] = last_tool_call_id
                last_tool_call_id = None

        return Sample(messages=messages, functions=json.loads(row["functions"]))

    def eval_sample(self, sample: Sample, rng):
        assert isinstance(sample, Sample)
        prompt = sample.messages[:2]
        assistant_msg = sample.messages[2]
        try:
            result = self.completion_fn(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                tools=[{"type": "function", "function": f} for f in sample.functions],
            )
            print("result", vars(result))
            sampled = result.get_completions()[0]
        except Exception as e:
            logging.info("Sampling failed!")
            logging.info(sample)
            logging.info(f"Prompt: {prompt}")
            logging.info(f"Error: {str(e)}")
            sampled = "ERROR: " + str(e)

        match = self.match_tool_call(assistant_msg, sampled)
        evals.record.record_metrics(score=match)

    def match_tool_call(self, gt_message: Sample, sampled: List[Union[str, Dict[str, Any]]]) -> int:
        sampled_tool_call = isinstance(sampled, dict) and sampled.get("type") == "function"
        expected_tool_call = gt_message.get("tool_calls") is not None

        if sampled_tool_call != expected_tool_call:
            return 0

        if not sampled_tool_call:
            # simply not using a tool call is considered a valid response
            return 1

        sampled_func = sampled["function"]
        gt_func = gt_message["tool_calls"][0]["function"]

        return (
            sampled_func["name"] == gt_func["name"]
            and sampled_func["arguments"] == gt_func["arguments"]
        )

    def run(self, recorder: RecorderBase):
        samples = self._get_dataset()
        self.eval_all_samples(recorder, samples)
        metrics = recorder.get_metrics()
        return {"score": np.mean([d["score"] for d in metrics])}

    def _get_dataset(self) -> List[Sample]:
        parsed = urlparse(self.dataset)
        query = parse_qs(parsed.query)
        query = {k: v[0] for k, v in query.items()}
        dataset = load_dataset(parsed.netloc + parsed.path, **query)
        return [self.load_sample(sample) for sample in dataset]
