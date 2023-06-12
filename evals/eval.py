"""
This file defines the base class for evals.
"""
import abc
import asyncio
import logging
import os
import random
from multiprocessing.pool import ThreadPool
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from tqdm import tqdm

from evals.api import CompletionFn

from .data import get_jsonl
from .record import RecorderBase
from .registry import Registry

logger = logging.getLogger(__name__)


SHUFFLE_SEED = 123
_MAX_SAMPLES = None


def _index_samples(samples: List[Any]) -> List[Tuple[Any, int]]:
    """Shuffle `samples` and pair each sample with its index."""
    indices = list(range(len(samples)))
    random.Random(SHUFFLE_SEED).shuffle(indices)
    if _MAX_SAMPLES is not None:
        indices = indices[:_MAX_SAMPLES]
    logger.info(f"Evaluating {len(indices)} samples")
    work_items = [(samples[i], i) for i in indices]
    return work_items


def set_max_samples(max_samples: int):
    global _MAX_SAMPLES
    _MAX_SAMPLES = max_samples


class Eval(abc.ABC):
    """
    Evaluation classes generally should override two methods:
    `eval_sample`: Takes in a test sample and a random number generator and
        records the metrics of interest.
    `run`: Takes in a recorder and runs the evaluation. Generally, most `run`
        methods will follow this same pattern: loading the data, calling
        `eval_all_samples`, and aggregating the recorded results.
    """

    def __init__(
        self,
        completion_fns: list[CompletionFn],
        seed: int = 20220722,
        name: str = "no_name_eval.default",
        registry: Optional[Registry] = None,
        samples_jsonl: Optional[str] = None,
    ):
        splits = name.split(".")
        if len(splits) < 2:
            raise ValueError(f"Eval name must at least have <base_eval>.<split>. Got name {name}")

        self.completion_fns = completion_fns
        self.seed = seed
        self.name = name
        self.registry = registry or Registry()
        self.samples_jsonl = samples_jsonl

    @abc.abstractmethod
    def eval_sample(self, sample: Any, rng: random.Random):
        raise NotImplementedError()

    @property
    def completion_fn(self) -> CompletionFn:
        """Helper for more ergonomic access to a single CompletionFn."""
        return self.completion_fns[0]

    @abc.abstractmethod
    def run(self, recorder: RecorderBase) -> Dict[str, float]:
        """Run the evaluation with the corresponding recorder."""
        raise NotImplementedError()

    async def async_eval_all_samples(
        self,
        eval_fn: Callable[[Tuple[Any, int]], Awaitable[Tuple[int, Any]]],
        samples: List[Any],
        concurrency: int = 32,
        show_progress: bool = True,
        **_kwargs: Any,
    ):
        work_items = _index_samples(samples)
        semaphore = asyncio.Semaphore(concurrency)

        async def eval_fn_with_semaphore(args):
            async with semaphore:
                return await eval_fn(args)

        futures = [asyncio.ensure_future(eval_fn_with_semaphore(args)) for args in work_items]

        for future in tqdm(
            asyncio.as_completed(futures), total=len(samples), disable=not show_progress
        ):
            await future

    def eval_all_samples(
        self,
        recorder: RecorderBase,
        samples,
        show_progress=True,
        record_raw_sample=True,
        **_kwargs: Any,
    ):
        """
        Evaluate all provided samples in parallel.
        """
        work_items = _index_samples(samples)
        threads = int(os.environ.get("EVALS_THREADS", "10"))
        show_progress = bool(os.environ.get("EVALS_SHOW_EVAL_PROGRESS", show_progress))

        def eval_sample(args):
            """
            Evaluate a single sample.
            """
            sample, idx = args
            base_name, split = self.name.split(".")[0:2]
            sample_id = f"{base_name}.{split}.{idx}"
            with recorder.as_default_recorder(sample_id):
                seed = f"{sample_id}:{self.seed}".encode("utf-8")
                rng = random.Random(seed)
                return idx, self.eval_sample(sample, rng)

        with ThreadPool(threads) as pool:
            if os.environ.get("EVALS_SEQUENTIAL", "0") in {"1", "true", "yes"}:
                logger.info(f"Running in sequential mode!")
                iter = map(eval_sample, work_items)
            else:
                logger.info(f"Running in threaded mode with {threads} threads!")
                iter = pool.imap_unordered(eval_sample, work_items)
            idx_and_result = list(tqdm(iter, total=len(work_items), disable=not show_progress))
        return [r for _, r in sorted(idx_and_result)]

    def get_samples(self):
        if self.samples_jsonl is None:
            raise ValueError(
                "To use `get_samples`, you must provide a `samples_jsonl` path." "Got `None`."
            )

        return get_jsonl(self.samples_jsonl)
