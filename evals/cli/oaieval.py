"""
This file defines the `oaieval` CLI for running evals.
"""
import argparse
import logging
import shlex
import sys
from functools import cached_property
from typing import Any, Mapping, Optional

import openai

import evals
import evals.api
import evals.base
import evals.record
from evals.base import ModelSpec, ModelSpecs
from evals.registry import Registry

logger = logging.getLogger(__name__)


def _purple(str):
    return f"\033[1;35m{str}\033[0m"


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run evals through the API")
    parser.add_argument("model", type=str, help="Name of a completion model.")
    parser.add_argument("eval", type=str, help="Name of an eval. See registry.")
    parser.add_argument("--embedding_model", type=str, default="")
    parser.add_argument("--ranking_model", type=str, default="")
    parser.add_argument("--extra_eval_params", type=str, default="")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visible", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--seed", type=int, default=20220722)
    parser.add_argument("--user", type=str, default="")
    parser.add_argument("--record_path", type=str, default=None)
    parser.add_argument(
        "--log_to_file", type=str, default=None, help="Log to a file instead of stdout"
    )
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--local-run", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run-logging", action=argparse.BooleanOptionalAction, default=True)
    return parser


def n_ctx_from_model_name(model_name: str) -> Optional[int]:
    """Returns n_ctx for a given API model name. Model list last updated 2023-03-14."""
    # note that for most models, the max tokens is n_ctx + 1
    DICT_OF_N_CTX_BY_MODEL_NAME_PREFIX: dict[str, int] = {
        "dummy-": 2048,
        "gpt-3.5-turbo-": 4096,
        "gpt-4-": 8192,
        "gpt-4-32k-": 32768,
    }
    DICT_OF_N_CTX_BY_MODEL_NAME: dict[str, int] = {
        "ada": 2048,
        "text-ada-001": 2048,
        "babbage": 2048,
        "text-babbage-001": 2048,
        "curie": 2048,
        "text-curie-001": 2048,
        "davinci": 2048,
        "text-davinci-001": 2048,
        "code-davinci-002": 8000,
        "text-davinci-002": 4096,
        "text-davinci-003": 4096,
        "gpt-3.5-turbo": 4096,
        "gpt-3.5-turbo-0301": 4096,
        "gpt-4": 8192,
        "gpt-4-0314": 8192,
        "gpt-4-32k": 32768,
        "gpt-4-32k-0314": 32768,
    }
    # first, look for a prefix match
    for model_prefix, n_ctx in DICT_OF_N_CTX_BY_MODEL_NAME_PREFIX.items():
        if model_name.startswith(model_prefix):
            return n_ctx
    # otherwise, look for an exact match and return None if not found
    return DICT_OF_N_CTX_BY_MODEL_NAME.get(model_name, None)


class ModelResolver:
    # This is a temporary method to identify which models are chat models.
    # Eventually, the OpenAI API should expose this information directly.
    CHAT_MODELS = {
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0301",
        "gpt-4",
        "gpt-4-0314",
        "gpt-4-32k",
        "gpt-4-32k-0314",
        "dummy-chat",
    }

    DUMMY_MODELS = {
        "dummy-chat",
        "dummy-completion",
    }

    def resolve(self, name: str) -> ModelSpec:
        if name in self.DUMMY_MODELS:
            result = ModelSpec(name=name, model=name, is_chat=(name in self.CHAT_MODELS))
            return result

        if name in self.api_model_ids:
            result = ModelSpec(
                name=name,
                model=name,
                is_chat=(name in self.CHAT_MODELS),
                n_ctx=n_ctx_from_model_name(name),
            )
            return result

        raise ValueError(f"Couldn't find model: {name}")

    @cached_property
    def api_model_ids(self):
        return [m["id"] for m in openai.Model.list()["data"]]


def run(args, registry: Optional[Registry] = None):
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    visible = args.visible if args.visible is not None else (args.max_samples is None)

    if args.max_samples is not None:
        evals.eval.set_max_samples(args.max_samples)

    registry = registry or Registry()
    eval_spec = registry.get_eval(args.eval)
    assert (
        eval_spec is not None
    ), f"Eval {args.eval} not found. Available: {list(sorted(registry._evals.keys()))}"

    model_resolver = ModelResolver()

    def get_model(name: str) -> ModelSpec:
        return model_resolver.resolve(name)

    completion_model_specs = [get_model(model) for model in args.model.split(",")]
    model_specs = ModelSpecs(
        completions_=completion_model_specs,
        embedding_=get_model(args.embedding_model) if args.embedding_model else None,
        ranking_=get_model(args.ranking_model) if args.ranking_model else None,
    )

    run_config = {
        "model_specs": model_specs,
        "eval_spec": eval_spec,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "command": " ".join(map(shlex.quote, sys.argv)),
        "initial_settings": {
            "visible": visible,
        },
    }

    model_name = model_specs.completions_[0].name if len(model_specs.completions_) > 0 else "n/a"
    eval_name = eval_spec.key
    run_spec = evals.base.RunSpec(
        model_name=model_name,
        model_names=model_specs.names,
        eval_name=eval_name,
        base_eval=eval_name.split(".")[0],
        split=eval_name.split(".")[1],
        run_config=run_config,
        created_by=args.user,
    )
    if args.record_path is None:
        record_path = f"/tmp/evallogs/{run_spec.run_id}_{args.model}_{args.eval}.jsonl"
    else:
        record_path = args.record_path
    if args.dry_run:
        recorder = evals.record.DummyRecorder(run_spec=run_spec, log=args.dry_run_logging)
    elif args.local_run:
        recorder = evals.record.LocalRecorder(record_path, run_spec=run_spec)
    else:
        recorder = evals.record.Recorder(record_path, run_spec=run_spec)

    api_extra_options = {}
    if not args.cache:
        api_extra_options["cache_level"] = 0

    run_url = f"{run_spec.run_id}"
    logger.info(_purple(f"Run started: {run_url}"))

    def parse_extra_eval_params(param_str: Optional[str]) -> Mapping[str, Any]:
        """Parse a string of the form "key1=value1,key2=value2" into a dict."""
        if not param_str:
            return {}

        def to_number(x):
            try:
                return int(x)
            except:
                pass
            try:
                return float(x)
            except:
                pass
            return x

        str_dict = dict(kv.split("=") for kv in param_str.split(","))
        return {k: to_number(v) for k, v in str_dict.items()}

    extra_eval_params = parse_extra_eval_params(args.extra_eval_params)

    eval_class = registry.get_class(eval_spec)
    eval = eval_class(
        model_specs=model_specs,
        seed=args.seed,
        name=eval_name,
        registry=registry,
        **extra_eval_params,
    )
    result = eval.run(recorder)
    recorder.record_final_report(result)

    if not (args.dry_run or args.local_run):
        logger.info(_purple(f"Run completed: {run_url}"))

    logger.info("Final report:")
    for key, value in result.items():
        logger.info(f"{key}: {value}")
    return run_spec.run_id


def main():
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    logging.basicConfig(
        format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
        filename=args.log_to_file if args.log_to_file else None,
    )
    logging.getLogger("openai").setLevel(logging.WARN)
    if hasattr(openai.error, "set_display_cause"):
        openai.error.set_display_cause()
    run(args)


if __name__ == "__main__":
    main()
