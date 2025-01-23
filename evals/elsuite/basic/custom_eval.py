import numpy as np
import evals
from evals.api import CompletionFn
from evals.record import RecorderBase
import sacrebleu
from rouge_score import rouge_scorer

class CustomEval(evals.Eval):
    def __init__(
            self,
            completion_fns: list[CompletionFn],
            samples_jsonl: str,
            *args,
            max_tokens: int = 100,
            **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert len(completion_fns) == 1, "CustomEval only supports one completion fn"
        self.max_tokens = max_tokens
        self.samples_jsonl = samples_jsonl
    
    def eval_sample(self, test_sample, rng):
        del rng

        assert isinstance(test_sample, dict), "sample must be a dict"
        assert "input" in test_sample, "sample must have an 'input' key"
        assert "ideal" in test_sample, "sample must have an 'ideal' key"

        prompt = test_sample["input"]
        ideal_answer = test_sample["ideal"]
        ideal_yesno = ideal_answer.split("\n")[0]

        result = self.completion_fn(
            prompt = prompt,
            temperature=0.0,
            max_tokens=self.max_tokens,
        )
        sampled = result.get_completions()[0]

        sampled_yesno = sampled.split("\n")[0]
        yesno_correct = sampled_yesno.strip() == ideal_yesno.strip()

        #Calculate BLEU Score
        bleu_score = sacrebleu.corpus_bleu([sampled], [[ideal_answer]]).score

        #Calculate ROUGE Score
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(ideal_answer, sampled)
        rougeL_f1_score = scores['rougeL'].fmeasure

        evals.record.record_match(
            yesno_correct,
            picked=sampled,
            sampled=sampled
        )

        evals.record.record_metrics(
            yesno_accuracy=float(yesno_correct),
            BLEU_score=float(bleu_score),
            ROUGE_L_score_f1=float(rougeL_f1_score)
        )
    
    def run(self, recorder: RecorderBase):
        samples = self.get_samples()
        self.eval_all_samples(recorder, samples)

        return{
            "yesno_accuracy": np.mean(recorder.get_scores("yesno_accuracy")),
            "BLEU_score": np.mean(recorder.get_scores("BLEU_score")),
            "ROUGE_L_score_f1": np.mean(recorder.get_scores("ROUGE_L_score_f1"))
        }
