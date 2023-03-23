"""
Generic eval that uses a prompt + classification.
"""
import itertools
import logging
import string
from collections import Counter
from random import Random
from typing import Callable, Iterable, Optional, Union

import openai

import evals
import evals.record
from evals.base import ModelSpec
from evals.elsuite.utils import PromptFn, format_necessary, scrub_formatting_from_prompt

INVALID_STR = "__invalid__"
CHOICE_KEY = "choice"
MATCH_FNS = {
    "include": lambda x, y: float(x in y),
    "exact": lambda x, y: float(x == y),
    "endswith": lambda x, y: x.endswith(y),
    "starts_or_endswith": lambda x, y: x.startswith(y) or x.endswith(y),
}

ANSWER_PROMPTS = {
    # e.g. "Yes"
    "classify": "Answer the question by printing only a single choice from {choices} (without quotes or punctuation) corresponding to the correct answer with no other text.".strip(),
    # e.g. "Yes\n The reasons are: ..."
    "classify_cot": "First, answer by printing a single choice from {choices} (without quotes or punctuation) corresponding to the correct answer. Then, from the next line, explain your reasonings step by step.".strip(),
    # e.g. "Let's think step by step. ...\nYes"
    "cot_classify": """
First, write out in a step by step manner your reasoning to be sure that your conclusion is correct. Avoid simply stating the correct answer at the outset. Then print only a single choice from {choices} (without quotes or punctuation) on its own line corresponding to the correct answer. At the end, repeat just the answer by itself on a new line.

Reasoning:""".strip(),
    "cot_classify_jp": """
まず、一歩一歩あなたの推論を書き出してください。単に正しい答えを最初に述べることを避けてください。次に、{choices}（引用符や句読点なし）から正しい答えに対応する1つの選択肢を単独の行に書きだしてください。最後に、答えだけを新しい行に繰り返してください。

推論：
    """.strip(),
}


def choice_to_str(choice_strings: Iterable[str]) -> str:
    """Return a string of choices, e.g. '"Yes" or "No" or "Maybe"'."""
    return " or ".join(f'"{choice}"' for choice in choice_strings)


def get_choice(text: str, eval_type: str, match_fn: Callable, choice_strings: Iterable[str]) -> str:
    """Clean the answer string to a choice string to one of choice_strings. Return '__invalid__.' if no match."""
    lines = text.strip().split("\n")
    if eval_type.startswith("cot_classify"):
        lines = lines[::-1]  # reverse lines
    for line in lines:
        line = line.strip()
        line = "".join(c for c in line if c not in string.punctuation)
        if not line:
            continue
        for choice in choice_strings:
            if match_fn(line, choice):
                return choice
    return INVALID_STR


def expand_args_dict(args_dict):
    """Expand a dict of dicts, with namings.

    args_dict = {
        "a": {"a1": 1, "a2": 2},
        "b": {"b1": 3, "b2": 4},
    }
    expand_args_dict(args_dict) = {
        "a=a1:b=b1": {"a": ("a1", 1), "b": ("b1", 3)},
        "a=a1:b=b2": {"a": ("a1", 1), "b": ("b2", 4)},
    ...}
    """
    args_dict = {k: list(v.items()) for k, v in args_dict.items()}
    keys = list(args_dict.keys())
    values = list(args_dict.values())
    new_values = [dict(zip(keys, v)) for v in itertools.product(*values)]
    new_names = [":".join([f"{k}={v[0]}" for k, v in sorted(d.items())]) for d in new_values]
    return dict(zip(new_names, new_values))


class ModelBasedClassify(evals.Eval):
    invalid_request_during_completion = 0
    invalid_request_during_evaluation = 0

    def __init__(
        self,
        model_specs: evals.ModelSpecs,
        samples_jsonl: str,
        modelgraded_spec_file: str,
        *args,
        match_fn: str = "starts_or_endswith",
        max_tokens: int = 1024,
        multicomp_n: Union[int, str] = 1,
        multicomp_temperature: float = 0.4,
        samples_renamings: Optional[dict[str, str]] = None,
        eval_type: Optional[str] = None,
        metaeval: bool = False,
        modelgraded_spec_args: Optional[dict[str, dict[str, str]]] = None,
        **kwargs,
    ):
        super().__init__(model_specs, *args, **kwargs)
        n_models = len(self.model_specs.completions)
        self.max_tokens = max_tokens
        self.samples_jsonl = samples_jsonl
        self.match_fn = MATCH_FNS[match_fn]
        self.metaeval = metaeval
        if multicomp_n == "from_models":
            assert n_models > 1, f"multicomp_n='from_models' but only 1 model is specified."
            self.multicomp_n = n_models
        else:
            assert isinstance(
                multicomp_n, int
            ), f"multicomp_n={multicomp_n} must be an int or 'from_models'."
            self.multicomp_n = multicomp_n
        self.multicomp_temperature = multicomp_temperature
        self.samples_renamings = samples_renamings or {}

        # check if multiple models are specified
        if len(self.model_specs.completions) > 1:
            assert (
                self.multicomp_n == n_models
            ), f"multicomp_n={self.multicomp_n} must be equal to the number of models={len(self.model_specs.completions)} if multiple models are specified."

        if self.model_spec.name == "dummy-completion" or self.model_spec.name == "dummy-chat":
            self.eval_modelspec = self.model_spec
        else:
            self.eval_modelspec = ModelSpec(
                name="gpt-3.5-turbo", model="gpt-3.5-turbo", is_chat=True
            )

        """import prompt and set attributes"""
        modelgraded_specs = self.registry.get_modelgraded_spec(modelgraded_spec_file)

        # 'choice_strings' is a list of strings that specifies the possible choices
        self.choice_strings = modelgraded_specs.pop("choice_strings")
        if self.choice_strings == "from_n":
            self.choice_strings = [str(i + 1) for i in range(self.multicomp_n)]
        elif self.choice_strings == "from_n_abc":
            self.choice_strings = [string.ascii_lowercase[i % 26] for i in range(self.multicomp_n)]
        elif self.choice_strings == "from_n_ABC":
            self.choice_strings = [string.ascii_uppercase[i % 26] for i in range(self.multicomp_n)]
        # make sure each choice doesn't contain any punctuation
        for s in self.choice_strings:
            assert not any(c in s for c in string.punctuation), f"{s} contains punctuation"
        #  (optional) 'choice_scores' is a dict that specifies the score for each choice string
        # if 'choice_scores' is specified, 'scores/' are computed and added to metrics
        self.choice_scores = modelgraded_specs.pop("choice_scores", {})
        if self.choice_scores == "from_strings":
            self.choice_scores = {c: float(c) for c in self.choice_strings}
        assert all(
            isinstance(v, (int, float)) for v in self.choice_scores.values()
        ), f"choice_scores must be a dict of floats, not {self.choice_scores}"

        # (optional) 'eval_type' is a string that specifies the type of classification algorithm
        #   - "classify": only answer
        #   - "cot_classify": reason then answer (chain-of-thought) <- most recommended
        #   - "classify_cot": answer then reason (explanation)
        # if 'eval_type' is not supplied from modelgraded_specs, then it must be supplied as an argument.
        #   - Importantly, it also assumes the answer prompt needs to be appended to the prompt.
        self.eval_type = modelgraded_specs.pop("eval_type", None)
        if not self.eval_type:
            append_answer_prompt = True  # append answer prompt to prompt
            assert (
                eval_type
            ), "eval_type must be specified, in modelgraded_spec_file or as an argument"
            self.eval_type = eval_type
        else:
            assert (
                not eval_type
            ), f"eval_type must be unspecified, if it is specified in modelgraded_spec_file"
            append_answer_prompt = False

        # 'prompt' is a string that specifies the model-graded evaluation
        prompt = modelgraded_specs.pop("prompt")
        assert isinstance(prompt, str), f"prompt must be a string, not {type(prompt)}"
        if append_answer_prompt:
            prompt += "\n\n" + ANSWER_PROMPTS[self.eval_type].format(
                choices=choice_to_str(self.choice_strings)
            )
        self.prompt = [{"role": "user", "content": prompt}]

        # 'input_outputs' is a dict that specifies the input and output keys in the sample
        # output key is the model's raw response to input key. These are used for filling 'prompt' template.
        self.input_outputs = modelgraded_specs.pop("input_outputs")
        assert isinstance(
            self.input_outputs, dict
        ), f"input_outputs must be a dict, not {type(self.input_outputs)}"

        # (optional) 'args' is a dict of dicts that specifies additional arguments for 'prompt'
        # each value in 'args_dict' essentially defines a separate modelgraded classification eval and has own metrics!
        # if 'modelgraded_spec_args' is specified in eval YAML, it is merged with 'args_dict'
        self.args_dict = modelgraded_specs.pop("args", {})
        self.args_dict.update(modelgraded_spec_args or {})
        if self.args_dict:
            self.expanded_args_dict = expand_args_dict(self.args_dict)
        else:
            self.expanded_args_dict = {}

        # (optional) 'completion_sample_templates'
        # each key must be one of 'input_outputs'.values(). If 'multicomp_n' > 1, this template is filled 'multicomp_n' times
        # and the concatenated result is passed to 'prompt' template.
        self.completion_sample_templates = modelgraded_specs.pop("completion_sample_templates", {})
        assert all(
            k in self.input_outputs.values() for k in self.completion_sample_templates
        ), f"all {self.completion_sample_templates.keys()} must be in {self.input_outputs.values()}, "
        if self.multicomp_n > 1:
            assert (
                self.completion_sample_templates
            ), "completion_sample_templates must be specified if multicomp_n > 1"

        # since we accept optional args, we need to check that all args are used
        for key in ("key", "group"):
            modelgraded_specs.pop(key, None)
        assert not modelgraded_specs, f"Unused args: {modelgraded_specs}. Typo in YAML?"

    def eval_sample(self, test_sample: dict, rng: Random) -> None:
        """Evaluate a single sample.

        Recorded metrics are always: one of the self.choice_strings, or "__invalid__".
        """
        if self.samples_renamings:
            test_sample = {self.samples_renamings.get(k, k): v for k, v in test_sample.items()}
        if self.multicomp_n > 1:
            test_sample["n"] = self.multicomp_n
        completions = {}
        if self.metaeval:
            # assert outputs exist in the data
            for v in self.input_outputs.values():
                assert v in test_sample, f"Missing output '{v}' in sample {test_sample.keys()}"
                completions[v] = test_sample[v]
        # remove outputs from the data
        test_sample = {
            k: v for k, v in test_sample.items() if k not in list(self.input_outputs.values())
        }
        for k in self.input_outputs:
            test_sample[k] = scrub_formatting_from_prompt(test_sample[k])

        if not self.metaeval:
            try:
                for k, v in self.input_outputs.items():
                    if self.multicomp_n > 1 and v in self.completion_sample_templates:
                        completion = ""
                        completion_i_template = self.completion_sample_templates[v]
                        for i in range(self.multicomp_n):
                            if len(self.model_specs.completions) > 1:
                                # use a separate model for each completion
                                model_spec = self.model_specs.completions[i]
                            else:
                                # use the single model for all completions
                                model_spec = self.model_spec
                            get_input_completion = PromptFn(
                                test_sample[k],
                                model_spec=model_spec,
                                max_tokens=self.max_tokens,
                                temperature=self.multicomp_temperature,
                            )
                            completion_i, _ = get_input_completion()
                            completion += format_necessary(
                                completion_i_template,
                                i=i + 1,
                                i_abc=string.ascii_lowercase[i % 26],
                                i_ABC=string.ascii_uppercase[i % 26],
                                output=completion_i,
                                n=self.multicomp_n,
                            )
                    else:
                        get_input_completion = PromptFn(
                            test_sample[k],
                            model_spec=self.model_spec,
                            max_tokens=self.max_tokens,
                        )
                        completion, _ = get_input_completion()
                    completions[v] = completion
            except openai.error.InvalidRequestError:
                self.invalid_request_during_completion += 1
                return

        try:
            metrics = {}
            evaluate = PromptFn(
                self.prompt,
                model_spec=self.eval_modelspec,
                max_tokens=self.max_tokens,
            )
            eval_kwargs = dict(**completions, **test_sample)
            if self.expanded_args_dict:
                args_dict = self.expanded_args_dict
            else:
                args_dict = {CHOICE_KEY: {}}
            for metric, args in args_dict.items():
                args = {k: v[1] for k, v in args.items()}
                evaluation, _ = evaluate(**args, **eval_kwargs)
                choice = get_choice(evaluation, self.eval_type, self.match_fn, self.choice_strings)
                if choice == INVALID_STR:
                    logging.warn(
                        f"Choices {self.choice_strings} not parsable for {self.eval_type}: {evaluation}"
                    )
                metrics[metric] = choice
                if self.metaeval:
                    assert (
                        metric in test_sample
                    ), f"Missing label for metric '{metric}' in sample {test_sample.keys()}"
                    metrics[metric + "_metascore"] = choice == test_sample[metric]

        except openai.error.InvalidRequestError:
            self.invalid_request_during_evaluation += 1
            return

        evals.record.record_metrics(**metrics)

        return choice

    def run(self, recorder):
        samples = evals.get_jsonl(self.samples_jsonl)

        self.eval_all_samples(recorder, samples)
        all_sample_metrics = recorder.get_metrics()

        record_metrics = {}
        if self.expanded_args_dict:
            metrics = sorted(self.expanded_args_dict)
        else:
            metrics = [CHOICE_KEY]
        for metric in metrics:
            chosen = [m[metric] for m in all_sample_metrics if metric in m]
            # if there is a best choice, compute the score
            if self.choice_scores:
                # assumption: each INVALID_STR contributes the lowest score
                lowest_score = min(self.choice_scores.values())
                scores = [
                    self.choice_scores[choice] if choice != INVALID_STR else lowest_score
                    for choice in chosen
                ]
                record_metrics[f"score/{metric}"] = sum(scores) / len(all_sample_metrics)
            # compute the counts and ratios
            counts = dict(Counter(chosen))
            missing_samples = len(all_sample_metrics) - len(chosen)
            if missing_samples:
                counts["__missing_samples__"] = missing_samples
            record_metrics.update({f"counts/{metric}/{k}": v for k, v in counts.items()})
            if self.metaeval:
                metascores = [m[metric + "_metascore"] for m in all_sample_metrics if metric in m]
                record_metrics[f"metascore/{metric}"] = sum(metascores) / len(all_sample_metrics)

        record_metrics["invalid_request_during_completion"] = self.invalid_request_during_completion
        record_metrics["invalid_request_during_evaluation"] = self.invalid_request_during_evaluation

        return record_metrics
