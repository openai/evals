from typing import Any

import numpy as np
from nltk import word_tokenize
from rouge_score import rouge_scorer
from scipy import interpolate

import evals
import evals.metrics


class ScicapSummarization(evals.Eval):
    def __init__(
        self,
        test_jsonl: str,
        rouge1_random_jsonl: str,
        rouge2_random_jsonl: str,
        rougeL_random_jsonl: str,
        max_tokens: int = 100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_tokens = max_tokens
        self.test_jsonl = test_jsonl
        self.metric_list = ["rouge1", "rouge2", "rougeL"]
        self.rouge = rouge_scorer.RougeScorer(
            self.metric_list,
            use_stemmer=True,
        )

        # build score normalization
        random_score_path = {
            "rouge1": rouge1_random_jsonl,
            "rouge2": rouge2_random_jsonl,
            "rougeL": rougeL_random_jsonl,
        }
        self.normalization_funcs = {}
        for metric in self.metric_list:
            random_data = evals.get_jsonl(random_score_path[metric])
            length_list = np.array([d["length"] for d in random_data])
            score_list = np.array([d["score"] for d in random_data])
            normalization_func = interpolate.interp1d(
                length_list,
                score_list,
                bounds_error=False,
                fill_value="extrapolate",
            )
            self.normalization_funcs[metric] = normalization_func

    def run(self, recorder):
        samples = evals.get_jsonl(self.test_jsonl)
        self.eval_all_samples(recorder, samples)
        events = recorder.get_events("match")

        # aggregate scores and normalize
        length = np.mean([e.data["length"] for e in events])
        aggregated_scores = {}
        for metric_name in self.metric_list:
            score = np.mean([e.data[metric_name] for e in events])
            score_normalized = score / self.normalization_funcs[metric_name](length)
            aggregated_scores[metric_name] = float(score)
            aggregated_scores[f"{metric_name}_normalized"] = float(score_normalized)

        return aggregated_scores

    def clean_text(self, text):
        return text.lower()

    def normalize_score(self, score):
        pass

    def eval_sample(self, sample: Any, *_):
        prompt = sample["input"]
        expected = sample["ideal"]

        # check
        assert prompt is not None, "prompt is None"
        assert expected is not None, "expected is None"
        assert isinstance(expected, str), "expected is not a string"

        # generate answer
        sampled = evals.sample_freeform(
            self.model_spec,
            prompt,
            max_tokens=self.max_tokens,
        )

        # preprocess
        expected = self.clean_text(expected)
        sampled = self.clean_text(sampled)

        # compute scores and match
        scores = self.rouge.score(expected, sampled)
        score_dict = {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
            "length": len(word_tokenize(sampled)),
        }
        match = score_dict["rouge2"] > 0.2  # not sure what to use here

        # record results
        evals.record.record_metrics(**score_dict)
        evals.record.record_match(
            correct=match,
            expected=expected,
            sampled=sampled,
            **score_dict,
        )
        return match
