from typing import Any

import evals
import evals.metrics
from evals.prompt.base import is_chat_prompt


import evals
import evals.elsuite.utils
import evals.metrics
import numpy as np

import re

class Generalized24Exec(evals.Eval):
    def __init__(
        self,
        model_specs: evals.ModelSpecs,
        samples_jsonl: str,
        *args,
        max_tokens: int = 500,
        **kwargs,
    ):
        super().__init__(model_specs, *args, **kwargs)
        self.max_tokens = max_tokens
        self.samples_jsonl = samples_jsonl

    def eval_sample(self, sample: Any, *_):
        sampled = evals.sample_freeform(
            self.model_spec, sample["input"], max_tokens=self.max_tokens
        )

        score = 0
        try:

            pattern = r'```(.*?)```'
            matches = re.findall(pattern, sampled, re.DOTALL)
            formula = matches[-1].strip().strip("\n").strip()

            lhs, rhs = formula.split("=")
            
            lhs, rhs = lhs.strip(), rhs.strip()
            print(lhs, rhs)
            gtlhs, gtrhs = sample['ideal'].split("=")
            gtlhs, gtrhs = gtlhs.strip(), gtrhs.strip()
            output = eval(lhs)

            

            pattern1 = r"[+\-*\\() ]"
            source_numbers = sorted([int(x) for x in re.split(pattern1, lhs) if len(x) > 0])
            gt_source_numbers = sorted([int(x) for x in re.split(pattern1, gtlhs) if len(x) > 0])

            score = 0
            if int(output) == int(rhs) \
                and int(rhs) == int(gtrhs.strip()) \
                    and source_numbers == gt_source_numbers:
                        print(source_numbers,gt_source_numbers)
                        score = 1
        except:
            pass


        # includes_answer = any(
        #     [evals.elsuite.utils.get_answer(sampled, ref) for ref in sample["ideal"]]
        # )
        evals.record.record_metrics(accuracy=float(score))
        return score

    def run(self, recorder):
        samples = evals.get_jsonl(self.samples_jsonl)
        self.eval_all_samples(recorder, samples)
        events = recorder.get_scores("accuracy")
        return {
            "accuracy": np.mean(events),
        }
