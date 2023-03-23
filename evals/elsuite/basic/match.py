from typing import Any

import evals
import evals.metrics
from evals.plugin.base import Plugin, PluginAction
from evals.prompt.base import is_chat_prompt


class Match(evals.Eval):
    def __init__(
        self,
        model_specs: evals.ModelSpecs,
        samples_jsonl: str,
        *args,
        max_tokens: int = 500,
        num_few_shot: int = 0,
        few_shot_jsonl: str = None,
        **kwargs,
    ):
        super().__init__(model_specs, *args, **kwargs)
        self.max_tokens = max_tokens
        self.samples_jsonl = samples_jsonl
        self.num_few_shot = num_few_shot
        if self.num_few_shot > 0:
            assert few_shot_jsonl is not None, "few shot requires few shot sample dataset"
            self.few_shot_jsonl = few_shot_jsonl
            self.few_shot = evals.get_jsonl(self.few_shot_jsonl)

    def eval_sample(self, sample: Any, *_):
        prompt = sample["input"]
        plugins = Plugin.load(sample.get("plugins", []))
        plugin_actions = PluginAction.load(sample.get("plugin_actions", []))

        if self.num_few_shot > 0:
            assert is_chat_prompt(sample["input"]), "few shot requires chat prompt"
            prompt = sample["input"][:-1]
            for s in self.few_shot[: self.num_few_shot]:
                prompt += s["sample"]
            prompt += sample["input"][-1:]

        return evals.check_sampled_text(
            self.model_spec,
            prompt,
            expected=sample["ideal"],
            plugins=plugins,
            plugin_actions=plugin_actions,
        )

    def run(self, recorder):
        samples = evals.get_jsonl(self.samples_jsonl)
        self.eval_all_samples(recorder, samples)
        events = recorder.get_events("match")
        return {
            "accuracy": evals.metrics.get_accuracy(events),
        }
