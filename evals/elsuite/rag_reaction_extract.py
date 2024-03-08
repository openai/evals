import json
import os
import re
import traceback
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel

import evals.metrics
from evals.api import CompletionFn
from evals.elsuite.rag_match import get_rag_dataset
from evals.elsuite.utils import ReactionDictMatching, ReactionDictMatchingSimple
from evals.record import RecorderBase, record_match

code_pattern = r"```[\s\S]*?\n([\s\S]+?)\n```"
json_pattern = r"```json[\s\S]*?\n([\s\S]+?)\n```"
csv_pattern = r"```csv[\s\S]*?\n([\s\S]+?)\n```"
table_pattern = r"\n({index0}[\s\S]+)\n[`]*"
outlink_pattern = r"\[Download[a-zA-Z0-9 ]+?\]\((https://[a-zA-Z0-9_. /]+?)\)"


class FileSampleWithInput(BaseModel):
    input: Optional[str]
    file_name: Optional[str]
    file_link: Optional[str]
    answerfile_name: Optional[str]
    answerfile_link: Optional[str]


class ReactionExtract(evals.Eval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        samples_jsonl: str,
        *args,
        instructions: Optional[str] = "",
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert len(completion_fns) < 3, "ReactionExtract only supports 3 completion fns"
        self.samples_jsonl = samples_jsonl
        self.instructions = instructions

    def eval_sample(self, sample, rng):
        assert isinstance(sample, dict)

        input_formatted = sample["input"] if type(sample["input"]) == list else [{"role": "user", "content": sample["input"]}]
        if self.instructions:
            prompt = [{"role": "system", "content": self.instructions}] + input_formatted
        else:
            prompt = input_formatted

        result = self.completion_fn(
            prompt=prompt,
            temperature=0.0,
            file_name=sample["file_name"],
            file_link=sample["file_link"]
        )
        sampled = result.get_completions()[0]
        correct_str = open(sample["answerfile_name"], 'r').read()
        correct_answer = json.loads(correct_str)
        # correct_answer = json.load(open(sample["answerfile_name"], 'r'))["inputs"]
        # correct_str = json.dumps(correct_answer, indent=4)

        try:
            if re.search(outlink_pattern, sampled) is not None:
                code = re.search(outlink_pattern, sampled).group()
                link = re.sub(outlink_pattern, r"\1", code)

                fname = f"/tmp/LLMEvals_{uuid.uuid4()}.json"
                os.system(f"wget {link} -O {fname}")
                answer = json.load(open(fname, 'r'))
            elif "json" in self.instructions:
                code = re.search(json_pattern, sampled).group()
                code_content = re.sub(json_pattern, r"\1", code)
                code_content = code_content.replace("\"", '"')

                # Delete comments
                code_content = re.sub(r'//.*', '', code_content)
                answer = json.loads(code_content)
            else:
                answer = {}
            picked_str = json.dumps(answer, indent=4)
            open(sample["file_name"].replace(".json", "_out.json"), 'w').write(picked_str)
        except:
            print(Path(sample["file_name"]).stem)
            traceback.print_exc()
            record_match(
                prompt=prompt,
                correct=False,
                expected=correct_str,
                picked=sampled,
                file_name=sample["file_name"],
                jobtype="match_all"
            )
            picked_str = "Failed to parse"
            answer = {}
            
        accuracy_leaves, df = ReactionDictMatchingSimple(correct_answer, answer)
        record_match(
            prompt=prompt,
            correct=(accuracy_leaves == 1.0),
            expected=correct_str,
            picked=picked_str,
            file_name=sample["file_name"],
            jobtype="match_all"
        )
        return {"accuracy_leaves": accuracy_leaves}

    def run(self, recorder: RecorderBase):
        samples = get_rag_dataset(self._prefix_registry_path(self.samples_jsonl).as_posix())
        metrics_all_sample = self.eval_all_samples(recorder, samples)

        metrics = {key: np.mean([sample_metrics[key] for sample_metrics in metrics_all_sample]) for key in metrics_all_sample[0].keys()}
        # if "SMILES" in raw_samples[0]["compare_fields"]:
        #     metrics["recall_SMILES"] = np.mean([sample_metrics["recall_SMILES"] for sample_metrics in metrics_all_sample
        #                                         if "recall_SMILES" in sample_metrics])
        return metrics
