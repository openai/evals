import asyncio
from pathlib import Path
from unittest.mock import Mock

import evals.eval as eval_module
from evals.api import DummyCompletionFn


class _AsyncProgressEval(eval_module.Eval):
    def eval_sample(self, sample, rng):
        raise NotImplementedError

    def run(self, recorder):
        raise NotImplementedError


def test_async_progress_total_uses_scheduled_work_count(monkeypatch):
    monkeypatch.setattr(eval_module, "_MAX_SAMPLES", 2)

    progress_totals = []

    def fake_tqdm(iterable, *, total, disable):
        progress_totals.append(total)
        return iterable

    monkeypatch.setattr(eval_module, "tqdm", fake_tqdm)

    evaluator = _AsyncProgressEval(
        completion_fns=[DummyCompletionFn()],
        eval_registry_path=Path("."),
        registry=Mock(),
    )
    completed = []

    async def eval_fn(item):
        completed.append(item)
        return item[1], None

    asyncio.run(
        evaluator.async_eval_all_samples(
            eval_fn,
            samples=list(range(5)),
            show_progress=True,
        )
    )

    assert len(completed) == 2
    assert progress_totals == [2]
