import os
from pathlib import Path
from typing import Any

import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider

import evals
import evals.metrics
from evals.api import CompletionFn
from evals.prompt.base import is_chat_prompt


def init_oss():
    """
    Initialize OSS client.
    """
    # Please set OSS_ACCESS_KEY_ID & OSS_ACCESS_KEY_SECRET in your environment variables.
    auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())

    # 设置 Endpoint
    endpoint = 'https://oss-cn-beijing.aliyuncs.com'

    # 设置 Bucket
    bucket_name = 'dp-filetrans-bj'
    bucket = oss2.Bucket(auth, endpoint, bucket_name)

    return bucket


def get_rag_dataset(samples_jsonl: str) -> list[dict]:
    bucket = init_oss()
    raw_samples = evals.get_jsonl(samples_jsonl)

    for raw_sample in raw_samples:
        for ftype in ["", "answer"]:
            if f"{ftype}file_name" not in raw_sample and f"{ftype}file_link" not in raw_sample:
                continue
            if f"{ftype}file_name" in raw_sample:
                oss_file = "changjunhan/" + os.path.basename(raw_sample[f"{ftype}file_name"])
                raw_sample[f"{ftype}file_link"] = "https://dp-filetrans-bj.oss-cn-beijing.aliyuncs.com/" + oss_file

                exists = bucket.object_exists(oss_file)
                if exists:
                    print(f"文件 {oss_file} 已存在于 OSS 中。")
                else:
                    # 上传文件
                    bucket.put_object_from_file(oss_file, raw_sample[f"{ftype}file_name"])
                    print(f"文件 {oss_file} 已上传到 OSS。")
            if f"{ftype}file_link" in raw_sample:
                local_file = raw_sample[f"{ftype}file_name"] if f"{ftype}file_name" in raw_sample else \
                    os.path.basename(raw_sample[f"{ftype}file_link"])
                oss_file = "changjunhan/" + os.path.basename(raw_sample[f"{ftype}file_link"])
                if not os.path.exists(local_file):
                    if bucket.object_exists(oss_file):
                        # 从 OSS 下载文件
                        Path(local_file).parent.mkdir(parents=True, exist_ok=True)
                        bucket.get_object_to_file(oss_file, local_file)
                        print(f"文件 {oss_file} 已下载到本地。")
    return raw_samples


class RAGMatch(evals.Eval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        samples_jsonl: str,
        *args,
        max_tokens: int = 500,
        num_few_shot: int = 0,
        few_shot_jsonl: str = None,
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert len(completion_fns) == 1, "Match only supports one completion fn"
        self.max_tokens = max_tokens
        self.samples_jsonl = samples_jsonl
        self.num_few_shot = num_few_shot
        if self.num_few_shot > 0:
            assert few_shot_jsonl is not None, "few shot requires few shot sample dataset"
            self.few_shot_jsonl = few_shot_jsonl
            self.few_shot = evals.get_jsonl(self._prefix_registry_path(self.few_shot_jsonl))

    def eval_sample(self, sample: Any, *_):
        assert isinstance(sample, dict), "sample must be a dict"
        assert "input" in sample, "sample must have an 'input' key"
        assert "ideal" in sample, "sample must have an 'ideal' key"
        assert isinstance(sample["ideal"], str) or isinstance(
            sample["ideal"], list
        ), "sample['ideal'] must be a string or list of strings"

        prompt = sample["input"]
        if self.num_few_shot > 0:
            assert is_chat_prompt(sample["input"]), "few shot requires chat prompt"
            prompt = sample["input"][:-1]
            for s in self.few_shot[: self.num_few_shot]:
                prompt += s["sample"]
            prompt += sample["input"][-1:]

        result = self.completion_fn(
            prompt=prompt,
            temperature=0.0,
            **{k: v for k, v in sample.items() if k not in ["input", "ideal"]}
        )
        sampled = result.get_completions()[0]

        return evals.record_and_check_match(
            prompt=prompt,
            sampled=sampled,
            expected=sample["ideal"],
            file_name=sample["file_name"]
        )

    def run(self, recorder):
        samples = get_rag_dataset(self._prefix_registry_path(self.samples_jsonl).as_posix())
        self.eval_all_samples(recorder, samples)
        events = recorder.get_events("match")
        return {
            "accuracy": evals.metrics.get_accuracy(events),
            "boostrap_std": evals.metrics.get_bootstrap_accuracy_std(events),
        }
