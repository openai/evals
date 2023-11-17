import csv
import re
from typing import Any, Mapping, Optional, Sequence

import blobfile as bf

import evals
import evals.metrics


class SageEval(evals.Eval):
    def get_csv(
        self, path: str, fieldnames: Optional[Sequence[str]] = None
    ) -> list[dict[str, Any]]:
        with bf.BlobFile(path, "r", cache_dir="/tmp/bf_cache", streaming=False) as f:
            reader = csv.DictReader(f, fieldnames=fieldnames)
            return [row for row in reader]

    def format_annotated_string(self, prompt: str, sample: Mapping[str, str]) -> str:
        def replace_annotation(match) -> str:
            annotation_id = match.group(1)
            return sample.get(
                annotation_id, match.group(0)
            )  # Default to the original match if ID not found

        return re.sub(r"\{(\w+)\|[^}]*\}", replace_annotation, prompt)
