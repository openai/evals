import random
import numpy as np
from typing import List, Dict
from .base import ModelSpecs
from .record import Recorder
from .metrics import get_accuracy, get_confusion_matrix, compute_precision, compute_recall, compute_f_score
from .eval import Eval

class SpamEmailEval(Eval):
    def __init__(self, model_specs: ModelSpecs, seed: int = 20220722, name: str = "spam_email.default"):
        super().__init__(model_specs, seed, name)

    def eval_sample(self, sample: Dict, rng: random.Random):
        prompt = sample["email"]
        expected = sample["spam"]

        response = self.model_spec.query(prompt)

        picked = 1 if response["spam_probability"] > 0.5 else 0

        return {
            "correct": picked == expected,
            "expected": expected,
            "picked": picked,
        }

    def run(self, samples: List[Dict], recorder: Recorder) -> Dict[str, float]:
        with recorder as rec:
            results = self.eval_all_samples(rec, samples)
        
        events = rec.get_events()

        accuracy = get_accuracy(events)
        confusion_matrix = get_confusion_matrix(events)
        precision = compute_precision(confusion_matrix)
        recall = compute_recall(confusion_matrix)
        f_score = compute_f_score(confusion_matrix)

        return {
            "accuracy": accuracy,
            "confusion_matrix": confusion_matrix,
            "precision": precision,
            "recall": recall,
            "f_score": f_score,
        }
