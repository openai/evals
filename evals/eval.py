"""
This file defines the base class for evals.
"""
import abc
import asyncio
import logging
import os
import random
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union

from tqdm import tqdm

from evals.api import CompletionFn

from .data import get_jsonl
from .record import RecorderBase
from .registry import Registry
from .solvers.solver import Solver
from .solvers.utils import maybe_wrap_with_compl_fn, maybe_wrap_with_solver

import weave

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
        completion_fns: list[Union[CompletionFn, Solver]],
        eval_registry_path: Path,
        seed: int = 20220722,
        name: str = "no_name_eval.default",
        registry: Optional[Registry] = None,
        samples_jsonl: Optional[str] = None,
    ):
        splits = name.split(".")
        if len(splits) < 2:
            raise ValueError(f"Eval name must at least have <base_eval>.<split>. Got name {name}")

        self.completion_fns = [maybe_wrap_with_compl_fn(fn) for fn in completion_fns]
        self.eval_registry_path = eval_registry_path
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

    def run(self, recorder: RecorderBase) -> Dict[str, float]:
        """Run the evaluation with the corresponding recorder."""
        print("Running eval", self.name)

        @weave.op()
        def yovaluate() -> Dict[str, Any]:
            return self._run_impl(recorder)

        res = yovaluate()

        print("Got result for eval", self.name, f"res={res}")
        return res

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
                logger.info("Running in sequential mode!")
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

        samples_path = self._get_samples_path()
        return get_jsonl(samples_path.as_posix())

    def _get_samples_path(self) -> Path:
        return self._prefix_registry_path(self.samples_jsonl)

    def _prefix_registry_path(self, data_path: str) -> Path:
        if os.path.isfile(data_path):
            return Path(data_path)

        return self.eval_registry_path / "data" / data_path


class SolverEval(Eval):
    """
    Compared to Eval, SolverEval supports a single completion_fn which must be
    a `Solver` type (see solvers/solver.py). The Solver is what we evaluate,
    and Eval code should interact with the Solver instead of the CompletionFn
    directly. A new Solver is created for each sample, and the Solver is passed
    to eval_sample. This allows Solvers to be stateful (e.g. have a memory)
    without interfering with other samples.

    Otherwise, this is the same as Eval and requires the same methods to be
    implemented:
    `eval_sample`: Takes in a Solver, a test sample, and a random number
        generator and records the metrics of interest.
    `run`: Takes in a recorder and runs the evaluation. Generally, most `run`
        methods will follow this same pattern: loading the data, calling
        `eval_all_samples`, and aggregating the recorded results.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (
            len(self.completion_fns) == 1
        ), f"{type(self).__name__} supports exactly one completion_fn, got {len(self.completion_fns)}."
        # Technically, instead of arg `completion_fns: list[CompletionFn]` we
        # should just have `solver: Solver` but we keep the args unchanged for
        # compatibility with the existing codebase.
        self._solver = maybe_wrap_with_solver(self.completion_fns[0])

    @abc.abstractmethod
    def eval_sample(self, solver: Solver, sample: Any, rng: random.Random) -> None:
        raise NotImplementedError()

    def eval_all_samples(
        self,
        recorder: RecorderBase,
        samples,
        show_progress=True,
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

                per_sample_solver = self._solver.copy()
                return idx, self.eval_sample(per_sample_solver, sample, rng)

        with ThreadPool(threads) as pool:
            if os.environ.get("EVALS_SEQUENTIAL", "0") in {"1", "true", "yes"}:
                logger.info("Running in sequential mode!")
                iter = map(eval_sample, work_items)
            else:
                logger.info(f"Running in threaded mode with {threads} threads!")
                iter = pool.imap_unordered(eval_sample, work_items)

            idx_and_result = []
            try:
                for result in tqdm(iter, total=len(work_items), disable=not show_progress):
                    idx_and_result.append(result)
            except KeyboardInterrupt:
                # "Gentle interrupt" allows us to stop early and still get results
                gentle_interrupt = os.environ.get("EVALS_GENTLE_INTERRUPT", "0") in {
                    "1",
                    "true",
                    "yes",
                }
                if gentle_interrupt:
                    logger.info("Evaluation stopped because of KeyboardInterrupt")
                    logger.info(
                        f"Report will be based on {len(idx_and_result)} out of the planned {len(work_items)} samples"
                    )
                else:
                    raise

        return [r for _, r in sorted(idx_and_result)]
