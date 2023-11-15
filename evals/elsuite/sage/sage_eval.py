import csv
from typing import Any, Mapping, Optional, Sequence

import blobfile as bf
import clusters

import evals
import evals.metrics


class SageEval(evals.Eval):
    def get_csv(
        self, path: str, fieldnames: Optional[Sequence[str]] = None
    ) -> list[dict[str, Any]]:
        path = clusters.localize_path(path)
        with bf.BlobFile(path, "r", cache_dir="/tmp/bf_cache", streaming=False) as f:
            reader = csv.DictReader(f, fieldnames=fieldnames)
            return [row for row in reader]

    def format_prompt(self, prompt: str, sample: Mapping[str, str]) -> str:
        return prompt.format(**sample)
