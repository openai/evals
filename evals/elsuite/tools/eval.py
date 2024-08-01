import json
import logging
from typing import Any, Dict, List, Union
from urllib.parse import parse_qs, urlparse

from datasets import load_dataset
from pydantic import BaseModel

import evals
import evals.metrics
from evals.api import CompletionFn
from evals.record import RecorderBase

logger = logging.getLogger(__name__)


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
        assert len(completion_fns) == 1, "Tools tasks only support one completion fn"
        self.dataset = dataset
        self.temperature = temperature
        self.max_tokens = max_tokens

    def load_sample(self, row) -> list[Sample]:
        # The FireFunction test data that we have doesn't have the right tool_call_ids, so
        # we need to fix them up here.
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
        # For some reason non-OpenAI solvers don't work with additional keys on prompts.
        for msg in prompt:
            del msg["tool_calls"]
            del msg["tool_call_id"]
        assistant_msg = sample.messages[2]
        try:
            result = self.completion_fn(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                tools=[{"type": "function", "function": f} for f in sample.functions],
            )
            sampled = result.get_completions()[0]
        except Exception as e:
            logging.info("Sampling failed!")
            logging.info(sample)
            logging.info(f"Prompt: {prompt}")
            logging.info(f"Error: {str(e)}")
            sampled = "ERROR: " + str(e)

        score = self.score_tool_call(assistant_msg, sampled)
        match = score == 1
        evals.record.record_match(match, expected=assistant_msg, sampled=sampled, score=score)
        return match

    def score_tool_call(self, gt_message: Sample, sampled: List[Union[str, Dict[str, Any]]]) -> int:
        sampled_tool_call = isinstance(sampled, dict) and sampled.get("type") == "function"
        expected_tool_call = gt_message.get("tool_calls") is not None

        if sampled_tool_call != expected_tool_call:
            print(f"tool call mismatch: {sampled_tool_call} != {expected_tool_call}")
            print(f"sampled: {sampled}")
            return 0

        if not sampled_tool_call:
            # simply not using a tool call when none is needed is a correct response
            return 1

        # 0.333 for getting any tool call, 0.333 for the right function name, 0.333 for the right args
        sampled_func = sampled["function"]
        sampled_name = sampled_func["name"]
        sampled_args = json.loads(sampled_func["arguments"])
        gt_func = gt_message["tool_calls"][0]["function"]
        gt_name = gt_func["name"]
        gt_args = json.loads(gt_func["arguments"])
        raw_score = 1
        if sampled_name == gt_name:
            raw_score += 1
        else:
            print(f"Function name mismatch: {sampled_name} != {gt_name}")
        if sampled_args == gt_args:
            raw_score += 1
        else:
            print(f"Function arguments mismatch: {sampled_args} != {gt_args}")
        return raw_score / 3

    def run(self, recorder: RecorderBase):
        samples = self._get_dataset()
        self.eval_all_samples(recorder, samples)
        events = recorder.get_events("match")
        score = sum(e.data["score"] for e in events) / len(events)
        return {"accuracy": evals.metrics.get_accuracy(events), "score": score}

    def _get_dataset(self) -> List[Sample]:
        parsed = urlparse(self.dataset)
        query = parse_qs(parsed.query)
        query = {k: v[0] for k, v in query.items()}
        dataset = load_dataset(parsed.netloc + parsed.path, **query)
        return [self.load_sample(sample) for sample in dataset]
