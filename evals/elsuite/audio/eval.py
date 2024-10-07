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
    get_audio_duration,
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
        max_audio_duration: int = 30,
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
        self.max_audio_duration = max_audio_duration

    @property
    def recorder(self):
        return self._recorder

    def eval_sample(self, sample: Sample, rng):
        prompt = self._build_prompt(sample, self.text_only)
        kwargs = self._get_completion_kwargs(sample)
        sampled = self._do_completion(prompt, **kwargs)
        return self._compute_metrics(sample, sampled)

    def _keep_sample(self, sample):
        """
        Allows for applying additional filtering to samples before evaluation.

        Currently only filters out samples with audio longer than max_audio_duration.
        """
        return get_audio_duration(sample["audio"]) < self.max_audio_duration

    def run(self, recorder: RecorderBase):
        x = recorder.record_sampling

        def record_sampling_wrapper(*args, **kwargs):
            messages = args[0]
            for msg in messages:
                if msg["role"] == "user":
                    redact_audio_content(msg["content"])
            x(*args, **kwargs)

        recorder.record_sampling = record_sampling_wrapper
        samples = self._load_dataset()
        samples = [s for s in samples if self._keep_sample(s)]
        self._recorder = recorder
        self.eval_all_samples(recorder, samples)
        return self._compute_corpus_metrics()

    def _load_dataset(self):
        ds = load_hf_dataset(self.dataset, evals.eval._MAX_SAMPLES).cast_column(
            "audio", Audio(sampling_rate=DEFAULT_SAMPLE_RATE)
        )
        return list(ds)

    def _get_completion_kwargs(self, sample: Sample):
        return {}

    def _do_completion(self, prompt, **kwargs):
        try:
            result = self.completion_fn(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **kwargs,
            )
            sampled = result.get_completions()[0]
        except Exception as e:
            for m in prompt:
                redact_audio_content(m["content"])
            logging.info("Sampling failed!")
            logging.info(f"Prompt: {prompt}")
            logging.info(f"Error: {str(e)}")
            sampled = "ERROR: " + str(e)
        return sampled


class MatchAudioTask(AudioTask):
    def _get_match_events(self):
        return self.recorder.get_events("match")

    def _get_expected_values(self):
        return [e.data["expected"] for e in self._get_match_events()]

    def _get_sampled_values(self):
        return [e.data["sampled"] for e in self._get_match_events()]

    def _compute_corpus_metrics(self):
        return {"accuracy": evals.metrics.get_accuracy(self._get_match_events())}


class ModelGradedAudioTask(AudioTask):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        dataset: str,
        eval_type: str,
        eval_completion_fn: CompletionFn,
        modelgraded_spec: str,
        modelgraded_spec_args: Optional[dict[str, dict[str, str]]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(completion_fns, dataset, *args, **kwargs)
        self.eval_type = eval_type
        self.eval_completion_fn = evals.registry.Registry().make_completion_fn(eval_completion_fn)
        self.eval_completion_kwargs = {"max_tokens": 1024}
        self.mg = self.registry.get_modelgraded_spec(modelgraded_spec)
        self.modelgraded_spec_args = modelgraded_spec_args or {}

    def _compute_metrics(self, sample: Sample, sampled: str):
        completions = {"completion": sampled}
        test_sample = {
            "input": self._build_prompt(sample, text_only=True),
            "ideal": self._get_expected(sample),
        }
        choice, info = classify(
            mg=self.mg,
            eval_type=self.eval_type,
            completion_fn=self.eval_completion_fn,
            completion_kwargs=self.eval_completion_kwargs,
            format_kwargs={**completions, **test_sample, **self.modelgraded_spec_args},
        )
        # "B" is a superset of the answer, "C" is a match, "E" is a match except for unimportant details.
        # As opposed to "A", a subset of the answer, or "D", a completely different answer.
        evals.record.record_metrics(choice=choice, score=1.0 if choice in "BCE" else 0.0)

    def _compute_corpus_metrics(self):
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

    def _build_prompt(self, sample: Sample, text_only: bool = False):
        input = sample["text"] if text_only else sample["audio"]
        return build_messages(self.DEFAULT_PROMPT, self.TASK_PROMPT, input)

    def _compute_metrics(self, sample: Sample, sampled):
        expected = sample["text"]
        score = self._compute_wer(expected, sampled)
        evals.record.record_metrics(wer=score)
        match = score < 0.1
        evals.record.record_match(match, expected=expected, sampled=sampled, wer=score)
        return match

    def _compute_corpus_metrics(self):
        metrics = super()._compute_corpus_metrics()
        metrics["wer"] = self._compute_wer(self._get_expected_values(), self._get_sampled_values())
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
    TASK_PROMPT = f"Please translate the text to {{language}}. Your response should only include the {{language}} translation, without any additional words:\n\n{AUDIO_PLACEHOLDER}"

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

    def _build_prompt(self, sample: Sample, text_only: bool = False):
        task_prompt = self.TASK_PROMPT.format(language=self.target_language)
        input = sample["sentence"] if text_only else sample["audio"]
        return build_messages(self.DEFAULT_PROMPT, task_prompt, input)

    def _compute_metrics(self, sample: Sample, sampled: str):
        expected = sample["translation"]
        score = self.bleu.sentence_score(sampled, [expected]).score
        evals.record.record_metrics(sacrebleu_sentence_score=score)
        match = score > 30
        if score is not None:
            evals.record.record_match(
                match, expected=expected, sampled=sampled, sacrebleu_sentence_score=score
            )
        return match

    def _compute_corpus_metrics(self):
        metrics = super()._compute_corpus_metrics()
        events = self._get_match_events()
        sampled = self._get_sampled_values()
        refs = [[e for e in self._get_expected_values()]]
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

    def _build_prompt(self, sample: Sample, text_only: bool = False):
        transcript = ""  # TODO: Add transcript
        input = transcript if text_only else sample["audio"]
        return build_messages(self.DEFAULT_PROMPT, self.TASK_PROMPT, input)

    def _compute_metrics(self, sample: Sample, sampled: str):
        expected = self.EMOTIONS[sample["label"]]
        normalized = sampled.strip().rstrip(string.punctuation).lower()
        match = normalized == expected
        evals.record.record_match(match, expected=expected, sampled=sampled)
        return match


class SpokenBoolQ(MatchAudioTask):
    TASK_PROMPT = f'Context:\n{{context}}\n\nAnswer the following question with only the single word "True" or "False", and no additional explanation: {AUDIO_PLACEHOLDER}'

    def _build_prompt(self, sample: Sample, text_only: bool = False):
        task_prompt = self.TASK_PROMPT.format(context=sample["passage"])
        input = sample["question"] if text_only else sample["audio"]
        return build_messages(self.DEFAULT_PROMPT, task_prompt, input)

    def _compute_metrics(self, sample: Sample, sampled: str):
        expected = str(sample["answer"]).lower()
        normalized = sampled.strip().split()[0].rstrip(string.punctuation).lower()
        match = normalized == expected
        evals.record.record_match(match, expected=expected, sampled=sampled)
        return match


class SpokenQA(ModelGradedAudioTask):
    TASK_PROMPT = f"Answer the following question: {AUDIO_PLACEHOLDER}"

    def _keep_sample(self, sample):
        """
        Questions with multiple answers are skipped, as it's not always clear if a correct
        answer needs to have any or all of the multiple answers.
        """
        return sample.get("answer") or len(sample.get("answers", [])) == 1

    def _build_prompt(self, sample: Sample, text_only: bool = False):
        input = sample["question"] if text_only else sample["audio"]
        return build_messages(self.DEFAULT_PROMPT, self.TASK_PROMPT, input)

    def _get_expected(self, sample: Sample):
        if sample.get("is_impossible"):
            return "the question is impossible to answer"
        if sample.get("answer"):
            return sample["answer"]
        answer = sample["answers"][0]
        return answer if isinstance(answer, str) else answer["text"]


class SpokenQAWithContext(SpokenQA):
    TASK_PROMPT = f"Context:\n{{context}}\n\nAnswer the following question: {AUDIO_PLACEHOLDER}"

    def _build_prompt(self, sample: Sample, text_only: bool = False):
        task_prompt = self.TASK_PROMPT.format(context=sample["context"])
        input = sample["question"] if text_only else sample["audio"]
        return build_messages(self.DEFAULT_PROMPT, task_prompt, input)


class SpokenTools(MatchAudioTask):
    def _load_dataset(self):
        def _make_samples(sample):
            # The FireFunction test data that we have doesn't have the right tool_call_ids, so
            # we need to fix them up here. We also remove spurious tool_calls and tool_call_ids
            # to prevent oaieval code from complaining.
            out = []
            last_tool_call_id = None
            user_audio_index = 0
            messages = sample["messages"]
            for i in range(len(messages)):
                m = messages[i]
                if m.get("tool_calls"):
                    last_tool_call_id = m["tool_calls"][-1]["id"]
                elif m.get("tool_call_id") is not None and last_tool_call_id is not None:
                    m["tool_call_id"] = last_tool_call_id
                    last_tool_call_id = None
                if m.get("tool_calls") is None:
                    m.pop("tool_calls")
                if m.get("tool_call_id") is None:
                    m.pop("tool_call_id")

                if i == len(messages) - 1 or (i > 0 and messages[i + 1]["role"] == "user"):
                    new_sample = sample.copy()
                    new_sample["messages"] = messages[: i + 1]
                    new_sample["user_message_audios"] = sample["user_message_audios"][
                        user_audio_index : user_audio_index + 1
                    ]
                    out.append(new_sample)
                    user_audio_index += 1
            return out

        # We first limit the dataset to MAX_SAMPLES to avoid loading the whole thing,
        # and then apply the limit again once we've generated the individual samples.
        ds = (
            load_hf_dataset(self.dataset, evals.eval._MAX_SAMPLES)
            .cast_column("user_message_audios", [Audio(sampling_rate=DEFAULT_SAMPLE_RATE)])
            .remove_columns(["conversation"])
        )
        samples = [sample for row in ds for sample in _make_samples(row)]
        if evals.eval._MAX_SAMPLES is not None:
            samples = samples[: evals.eval._MAX_SAMPLES]
        return samples

    def _keep_sample(self, sample):
        # If a sample is missing or is too long, we skip it.
        audios = sample.get("user_message_audios")
        keep = audios and get_audio_duration(audios[0]) < self.max_audio_duration
        if not keep:
            print("Skipping sample", sample)
        return keep

    def _build_prompt(self, sample: Sample, text_only: bool = False):
        # Remove all messages after the final user message,
        # and insert the audio content into that final user message, if needed.
        messages = sample["messages"].copy()
        while messages[-1]["role"] != "user":
            messages.pop()
        if not text_only:
            audio = sample["user_message_audios"][0]["array"]
            messages[-1]["content"] = make_audio_content(AUDIO_PLACEHOLDER, audio)
        return messages

    def _get_completion_kwargs(self, sample: Sample):
        functions = json.loads(sample["functions"])
        tools = [{"type": "function", "function": f} for f in functions]
        return {"tools": tools}

    def _compute_metrics(self, sample: Sample, sampled: str):
        # Get the first response to the user message.
        for msg in reversed(sample["messages"]):
            if msg["role"] == "user":
                break
            expected = msg
        score = self._score_tool_call(expected, sampled)
        match = score == 1
        evals.record.record_match(match, expected=expected, sampled=sampled, score=score)
        return match

    def _score_tool_call(
        self, gt_message: Sample, sampled: List[Union[str, Dict[str, Any]]]
    ) -> int:
        sampled_func = sampled.get("function") if isinstance(sampled, dict) else None
        expected_func = (
            gt_message.get("tool_calls")[0]["function"] if gt_message.get("tool_calls") else None
        )
        sampled_name = (sampled_func or {}).get("name")
        expected_name = (expected_func or {}).get("name")

        if bool(sampled_func) != bool(expected_func):
            print(f"Tool call mismatch, sampled: {sampled_name}, expected: {expected_name}")
            return 0

        if not sampled_func:
            # simply not using a tool call when none is needed is a correct response
            return 1

        # 0.333 for getting any tool call, 0.333 for the right function name, 0.333 for the right args
        raw_score = 1
        if sampled_name == expected_name:
            raw_score += 1
        else:
            print(f"Function name mismatch, sampled: {sampled_name}, expected: {expected_name}")
        sampled_args = json.loads(sampled_func["arguments"])
        expected_args = json.loads(expected_func["arguments"])
        if sampled_args == expected_args:
            raw_score += 1
        else:
            print(
                f"Function arguments mismatch, sampled: {sampled_args}, expected: {expected_args}"
            )
        return raw_score / 3

    def _compute_corpus_metrics(self):
        metrics = super()._compute_corpus_metrics()
        events = self._get_match_events()
        metrics["score"] = sum(e.data["score"] for e in events) / len(events)
        return metrics


class SpokenCompare(MatchAudioTask):
    TASK_PROMPT = f"{{instruction}}\nOnly respond with a number.\n1. <|reserved_special_token_0|>\n2. <|reserved_special_token_0|>{AUDIO_PLACEHOLDER}{AUDIO_PLACEHOLDER}"
    # TASK_PROMPT = f"{{instruction}}\nYour answer should only be a number.\n1. {AUDIO_PLACEHOLDER}\n2. {AUDIO_PLACEHOLDER}"

    def _load_dataset(self):
        ds = (
            load_hf_dataset(self.dataset, evals.eval._MAX_SAMPLES)
            .cast_column("audio", Audio(sampling_rate=DEFAULT_SAMPLE_RATE))
            .cast_column("audio2", Audio(sampling_rate=DEFAULT_SAMPLE_RATE))
        )
        return list(ds)

    def _build_prompt(self, sample: Sample, text_only: bool = False):
        task_prompt = self.TASK_PROMPT.format(instruction=sample["instruction"])
        audios = [sample["audio"], sample["audio2"]]
        return build_messages(self.DEFAULT_PROMPT, task_prompt, audios)

    def _compute_metrics(self, sample: Sample, sampled: str):
        expected = sample["label"]
        normalized = sampled.strip()[0]
        match = normalized == expected
        evals.record.record_match(match, expected=expected, sampled=sampled)
        return match
