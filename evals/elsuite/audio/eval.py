import json
import logging
import string
from collections import Counter
from typing import Any, Dict, List, Optional, Union

import jiwer
from datasets import Audio
from sacrebleu.metrics.bleu import BLEU

import evals
import evals.metrics
from evals.api import CompletionFn
from evals.elsuite.audio.utils import (
    AUDIO_PLACEHOLDER,
    DEFAULT_SAMPLE_RATE,
    build_messages,
    load_hf_dataset,
    make_audio_content,
    redact_audio_content,
)
from evals.elsuite.modelgraded.classify_utils import classify
from evals.record import RecorderBase

logger = logging.getLogger(__name__)

# This will typically correspond to a row in a HF dataset, although
# in some cases a row may produce multiple samples.
Sample = Dict[str, Any]


class AudioTask(evals.Eval):
    DEFAULT_PROMPT = "You are a helpful assistant."

    def __init__(
        self,
        completion_fns: list[CompletionFn],
        dataset: str,
        text_only: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        *args,
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert len(completion_fns) == 1, "Audio tasks only support one completion fn"
        self.dataset = dataset
        self.text_only = text_only
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._recorder = None

    @property
    def recorder(self):
        return self._recorder

    def eval_sample(self, sample: Sample, rng):
        assert isinstance(sample, Sample)
        prompt = self.build_prompt(sample, self.text_only)
        kwargs = self.get_completion_kwargs(sample)
        sampled = self.do_completion(prompt, **kwargs)
        return self.compute_metrics(sample, sampled)

    def run(self, recorder: RecorderBase):
        samples = self.load_dataset()
        self._recorder = recorder
        self.eval_all_samples(recorder, samples)
        return self.compute_corpus_metrics()

    def load_dataset(self):
        ds = load_hf_dataset(self.dataset, evals.eval._MAX_SAMPLES).cast_column(
            "audio", Audio(sampling_rate=DEFAULT_SAMPLE_RATE)
        )
        return [Sample(data=row) for row in ds]

    def get_completion_kwargs(self, sample: Sample):
        return {}

    def do_completion(self, prompt, **kwargs):
        try:
            result = self.completion_fn(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **kwargs,
            )
            sampled = result.get_completions()[0]
        except Exception as e:
            redact_audio_content(m["content"] for m in prompt)
            logging.info("Sampling failed!")
            logging.info(f"Prompt: {prompt}")
            logging.info(f"Error: {str(e)}")
            sampled = "ERROR: " + str(e)
        return sampled


class MatchAudioTask(AudioTask):
    def get_match_events(self):
        return self.recorder.get_events("match")

    def get_expected_values(self):
        return [e.data["expected"] for e in self.get_match_events()]

    def get_sampled_values(self):
        return [e.data["sampled"] for e in self.get_match_events()]

    def compute_corpus_metrics(self):
        return {"accuracy": evals.metrics.get_accuracy(self.get_match_events())}


class ModelGradedAudioTask(AudioTask):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        dataset: str,
        eval_completion_fn: CompletionFn,
        modelgraded_spec: str,
        modelgraded_spec_args: Optional[dict[str, dict[str, str]]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(completion_fns, dataset, *args, **kwargs)
        self.eval_completion_fn = evals.registry.Registry().make_completion_fn(eval_completion_fn)
        self.eval_completion_kwargs = {"max_tokens": 1024}
        self.mg = self.registry.get_modelgraded_spec(modelgraded_spec)
        self.modelgraded_spec_args = modelgraded_spec_args or {}

    def compute_metrics(self, sample: Sample, sampled: str):
        completions = {"completion": sampled}
        test_sample = {"input": self.build_prompt(sample, text_only=True)}
        # TODO(juberti): this needs to call get_sample to get the GT answer...
        choice, info = classify(
            mg=self.mg,
            completion_fn=self.eval_completion_fn,
            completion_kwargs=self.eval_completion_kwargs,
            format_kwargs={**completions, **test_sample, **self.modelgraded_spec_args},
        )
        evals.record.record_metrics(choice=choice, score=info["score"])

    def compute_corpus_metrics(self):
        metrics = {}
        events = self.recorder.get_metrics()
        choices = [m["choice"] for m in events]
        counts = dict(Counter(choices))
        metrics.update({f"counts/{k}": v for k, v in counts.items()})
        scores = [m["score"] for m in events if m["score"] is not None]
        if scores:
            metrics["accuracy"] = sum(scores) / len(scores)
        return metrics


class Transcribe(MatchAudioTask):
    TASK_PROMPT = f"Repeat the following text, without any explanation: {AUDIO_PLACEHOLDER}"

    def build_prompt(self, sample: Sample, text_only: bool = False):
        input = sample["text"] if text_only else sample["audio"]
        return build_messages(self.DEFAULT_PROMPT, self.TASK_PROMPT, input)

    def compute_metrics(self, sample: Sample, sampled):
        expected = sample["text"]
        score = self._compute_wer(expected, sampled)
        evals.record.record_metrics(wer=score)
        match = score < 0.1
        evals.record.record_match(match, expected=expected, sampled=sampled, wer=score)
        return match

    def compute_corpus_metrics(self):
        metrics = super().compute_corpus_metrics()
        metrics["wer"] = self._compute_wer(self.get_expected_values(), self.get_sampled_values())
        return metrics

    def _compute_wer(self, expected, sampled):
        transform = jiwer.Compose(
            [
                jiwer.RemovePunctuation(),
                jiwer.ToLowerCase(),
                jiwer.RemoveWhiteSpace(replace_by_space=True),
                jiwer.Strip(),
                jiwer.ReduceToListOfListOfWords(),
            ]
        )
        output = jiwer.process_words(
            expected, sampled, reference_transform=transform, hypothesis_transform=transform
        )
        return output.wer


class Translate(MatchAudioTask):
    TASK_PROMPT = f"Translate the following text into {{language}}, without any explanation: {AUDIO_PLACEHOLDER}"

    def __init__(
        self,
        completion_fns: list[CompletionFn],
        dataset: str,
        target_language: str,
        *args,
        **kwargs,
    ):
        super().__init__(completion_fns, dataset, *args, **kwargs)
        self.target_language = target_language
        self.bleu = BLEU(effective_order=True)

    def build_prompt(self, sample: Sample, text_only: bool = False):
        task_prompt = self.TASK_PROMPT.format(language=self.target_language)
        input = sample["audio"] if text_only else sample["sentence"]
        return build_messages(self.DEFAULT_PROMPT, task_prompt, input)

    def compute_metrics(self, sample: Sample, sampled: str):
        expected = sample["sentence"]
        score = self.bleu.sentence_score(sampled, [expected]).score
        evals.record.record_metrics(sacrebleu_sentence_score=score)
        match = score > 30
        if score is not None:
            evals.record.record_match(
                match, expected=expected, sampled=sampled, sacrebleu_sentence_score=score
            )
        return match

    def compute_corpus_metrics(self):
        metrics = super().compute_corpus_metrics()
        events = self.get_match_events()
        sampled = self.get_sampled_values()
        refs = [[e for e in self.get_expected_values()]]
        metrics["sacrebleu_score"] = self.bleu.corpus_score(sampled, refs).score
        metrics["sacrebleu_sentence_score"] = sum(
            e.data["sacrebleu_sentence_score"] for e in events
        ) / len(events)
        return metrics


class SpokenER(MatchAudioTask):
    # Note, this is specific to the individual SpokenER dataset and a more general approach will be needed.
    EMOTIONS = [
        "anger",
        "joy",
        "neutral",
        "sadness",
    ]
    TASK_PROMPT = f"Respond in a single word which of these emotions ({', '.join(EMOTIONS)}) best matches the following: {AUDIO_PLACEHOLDER}"

    def build_prompt(self, sample: Sample, text_only: bool = False):
        transcript = ""  # TODO: Add transcript
        input = transcript if text_only else sample["audio"]
        return build_messages(self.DEFAULT_PROMPT, self.TASK_PROMPT, input)

    def compute_metrics(self, sample: Sample, sampled: str):
        expected = self.EMOTIONS[sample["label"]]
        normalized = sampled.strip().rstrip(string.punctuation).lower()
        match = normalized == expected
        evals.record.record_match(match, expected=expected, sampled=sampled)
        return match


class SpokenBoolQ(MatchAudioTask):
    TASK_PROMPT = f'Context:\n{{context}}\n\nAnswer the following question with only the single word "True" or "False", and no additional explanation: {AUDIO_PLACEHOLDER}'

    def build_prompt(self, sample: Sample, text_only: bool = False):
        task_prompt = self.TASK_PROMPT.format(context=sample["passage"])
        input = sample["audio"] if text_only else sample["question"]
        return build_messages(self.DEFAULT_PROMPT, task_prompt, input)

    def compute_metrics(self, sample: Sample, sampled: str):
        expected = str(sample["answer"]).lower()
        normalized = sampled.strip().split()[0].rstrip(string.punctuation).lower()
        match = normalized == expected
        evals.record.record_match(match, expected=expected, sampled=sampled)
        return match


class SpokenQA(ModelGradedAudioTask):
    TASK_PROMPT = f"Context:\n{{context}}\n\nAnswer the following question: {AUDIO_PLACEHOLDER}"

    def build_prompt(self, sample: Sample, text_only: bool = False):
        task_prompt = self.TASK_PROMPT.format(context=sample["context"])
        input = sample["question"] if text_only else sample["audio"]
        return build_messages(self.DEFAULT_PROMPT, task_prompt, input)

    def get_expected(self, sample: Sample):
        return (
            sample["answers"][0]["text"]
            if not sample["is_impossible"]
            else "the question is impossible to answer"
        )


class SpokenTools(MatchAudioTask):
    def load_dataset(self):
        # TODO: create samples for each user message in the row.
        ds = load_hf_dataset(self.dataset).cast_column(
            "user_message_audios", [Audio(sampling_rate=DEFAULT_SAMPLE_RATE)]
        )
        return [Sample(data=row) for row in ds]

    def build_prompt(self, sample: Sample, text_only: bool = False):
        # The FireFunction test data that we have doesn't have the right tool_call_ids, so
        # we need to fix them up here. We also need to remove tool_calls from prompts for
        # non-OpenAI solvers.
        messages = sample["messages"]
        last_tool_call_id = None
        for m in messages:
            if m["tool_calls"]:
                last_tool_call_id = m["tool_calls"][-1]["id"]
            elif m.get("tool_call_id") is not None and last_tool_call_id is not None:
                m["tool_call_id"] = last_tool_call_id
                last_tool_call_id = None
            elif not m.get("tool_calls"):
                del m["tool_calls"]
                del m["tool_call_id"]
        messages = messages[:2]
        if not text_only:
            audio = sample["user_message_audios"][0]["array"]
            messages[1]["content"] = make_audio_content(AUDIO_PLACEHOLDER, audio)
        return messages[:2]

    def get_completion_kwargs(self, sample: Sample):
        functions = json.loads(sample["functions"])
        tools = [{"type": "function", "function": f} for f in functions]
        return {"tools": tools}

    def compute_metrics(self, sample: Sample, sampled: str):
        expected = sample["messages"][2]
        score = self.score_tool_call(expected, sampled)
        match = score == 1
        evals.record.record_match(match, expected=expected, sampled=sampled, score=score)
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

    def compute_corpus_metrics(self):
        metrics = super().compute_corpus_metrics()
        events = self.get_match_events()
        metrics["score"] = sum(e.data["score"] for e in events) / len(events)
        return metrics
