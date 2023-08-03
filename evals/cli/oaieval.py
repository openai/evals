"""
This file defines the `oaieval` CLI for running evals.
"""
import argparse
import logging
import shlex
import sys
from typing import Any, Mapping, Optional, Union, cast

import openai

import evals
import evals.api
import evals.base
import evals.record
from evals.eval import Eval
from evals.registry import Registry

logger = logging.getLogger(__name__)


def _purple(str: str) -> str:
    return f"\033[1;35m{str}\033[0m"


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run evals through the API")
    parser.add_argument(
        "completion_fn",
        type=str,
        help="One or more CompletionFn URLs, separated by commas (,). A CompletionFn can either be the name of a model available in the OpenAI API or a key in the registry (see evals/registry/completion_fns).",
    )
    parser.add_argument("eval", type=str, help="Name of an eval. See registry.")
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
    parser.add_argument(
        "--registry_path",
        type=str,
        default=None,
        action="append",
        help="Path to the registry",
    )
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--local-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable local mode for running evaluations. In this mode, the evaluation results are stored locally in a JSON file. This mode is enabled by default.",
    )

    parser.add_argument(
        "--http-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable HTTP mode for running evaluations. In this mode, the evaluation results are sent to a specified URL rather than being stored locally or in Snowflake. This mode should be used in conjunction with the '--http-run-url' and '--http-batch-size' arguments.",
    )

    parser.add_argument(
        "--http-run-url",
        type=str,
        default=None,
        help="URL to send the evaluation results when in HTTP mode. This option should be used in conjunction with the '--http-run' flag.",
    )

    parser.add_argument(
        "--http-batch-size",
        type=int,
        default=100,
        help="Number of events to send in each HTTP request when in HTTP mode. Default is 1, i.e., send events individually. Set to a larger number to send events in batches. This option should be used in conjunction with the '--http-run' flag.",
    )
    parser.add_argument(
        "--http-fail-percent-threshold",
        type=int,
        default=5,
        help="The acceptable percentage threshold of HTTP requests that can fail. Default is 5, meaning 5% of total HTTP requests can fail without causing any issues. If the failure rate goes beyond this threshold, suitable action should be taken or the process will be deemed as failing, but still stored locally.",
    )

    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run-logging", action=argparse.BooleanOptionalAction, default=True)
    return parser


class OaiEvalArguments(argparse.Namespace):
    completion_fn: str
    eval: str
    extra_eval_params: str
    max_samples: Optional[int]
    cache: bool
    visible: Optional[bool]
    seed: int
    user: str
    record_path: Optional[str]
    log_to_file: Optional[str]
    registry_path: Optional[str]
    debug: bool
    local_run: bool
    http_run: bool
    http_run_url: Optional[str]
    http_batch_size: int
    http_fail_percent_threshold: int
    dry_run: bool
    dry_run_logging: bool


def run(args: OaiEvalArguments, registry: Optional[Registry] = None) -> str:
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    visible = args.visible if args.visible is not None else (args.max_samples is None)

    if args.max_samples is not None:
        evals.eval.set_max_samples(args.max_samples)

    registry = registry or Registry()
    if args.registry_path:
        registry.add_registry_paths(args.registry_path)

    eval_spec = registry.get_eval(args.eval)
    assert (
        eval_spec is not None
    ), f"Eval {args.eval} not found. Available: {list(sorted(registry._evals.keys()))}"

    completion_fns = args.completion_fn.split(",")
    completion_fn_instances = [registry.make_completion_fn(url) for url in completion_fns]

    run_config = {
        "completion_fns": completion_fns,
        "eval_spec": eval_spec,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "command": " ".join(map(shlex.quote, sys.argv)),
        "initial_settings": {
            "visible": visible,
        },
    }

    eval_name = eval_spec.key
    if eval_name is None:
        raise Exception("you must provide a eval name")

    run_spec = evals.base.RunSpec(
        completion_fns=completion_fns,
        eval_name=eval_name,
        base_eval=eval_name.split(".")[0],
        split=eval_name.split(".")[1],
        run_config=run_config,
        created_by=args.user,
    )
    if args.record_path is None:
        record_path = f"/tmp/evallogs/{run_spec.run_id}_{args.completion_fn}_{args.eval}.jsonl"
    else:
        record_path = args.record_path

    if args.http_run:
        args.local_run = False
    elif args.local_run:
        args.http_run = False

    recorder: evals.record.RecorderBase
    recorder_kwargs = []
    if args.dry_run:
        recorder_class = evals.record.DummyRecorder
        recorder_args = {"run_spec": run_spec, "log": args.dry_run_logging}
    elif args.local_run:
        recorder_class = evals.record.LocalRecorder
        recorder_args = {"run_spec": run_spec}
        recorder_kwargs = [record_path]
    elif args.http_run:
        if args.http_run_url is None:
            raise ValueError("URL must be specified when using http-run mode")
        recorder_class = evals.record.HttpRecorder
        recorder_args = {
            "url": args.http_run_url,
            "run_spec": run_spec,
            "batch_size": args.http_batch_size,
            "fail_percent_threshold": args.http_fail_percent_threshold,
            "local_fallback_path": record_path,
        }

    else:
        recorder_class = evals.record.Recorder
        recorder_args = {"run_spec": run_spec}
        recorder_kwargs = [record_path]

    recorder = recorder_class(*recorder_kwargs, **recorder_args)

    api_extra_options: dict[str, Any] = {}
    if not args.cache:
        api_extra_options["cache_level"] = 0

    run_url = f"{run_spec.run_id}"
    logger.info(_purple(f"Run started: {run_url}"))

    def parse_extra_eval_params(
        param_str: Optional[str],
    ) -> Mapping[str, Union[str, int, float]]:
        """Parse a string of the form "key1=value1,key2=value2" into a dict."""
        if not param_str:
            return {}

        def to_number(x: str) -> Union[int, float, str]:
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
    eval: Eval = eval_class(
        completion_fns=completion_fn_instances,
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


def main() -> None:
    parser = get_parser()
    args = cast(OaiEvalArguments, parser.parse_args(sys.argv[1:]))
    logging.basicConfig(
        format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
        filename=args.log_to_file if args.log_to_file else None,
    )
    logging.getLogger("openai").setLevel(logging.WARN)

    if hasattr(openai.error, "set_display_cause"):  # type: ignore
        openai.error.set_display_cause()  # type: ignore
    run(args)


if __name__ == "__main__":
    main()
