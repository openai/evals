"""
This file defines the base specifications for models, evals, and runs. Running
evals and most development work should not require familiarity with this file.
"""
import base64
import datetime
import os
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Sequence

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass


@dataclass
class ModelSpec:
    """
    Specification for a model.
    """

    name: str
    model: Optional[str] = None

    is_chat: bool = False

    encoding: Optional[str] = None
    organization: Optional[str] = None
    api_key: Optional[str] = None
    extra_options: Optional[Mapping[str, Any]] = None
    headers: Optional[Mapping[str, Any]] = None
    strip_completion: bool = True
    n_ctx: Optional[int] = None
    format: Optional[str] = None
    key: Optional[str] = None
    group: Optional[str] = None

    def __post_init__(self):
        if self.extra_options is None:
            self.extra_options = {}
        if self.headers is None:
            self.headers = {}

        if self.model is None:
            raise ValueError(f"Must specify a model")


@dataclass
class BaseEvalSpec:
    """
    Specification for a base eval.
    """

    id: Optional[str] = None
    metrics: Optional[Sequence[str]] = None
    description: Optional[str] = None
    disclaimer: Optional[str] = None

    """
    True if higher values are better, False if lower values are better.
    This should really be part of a metric, but it's easier to put it here.
    """
    higher_is_better: bool = True

    key: Optional[str] = None
    group: Optional[str] = None


@dataclass
class EvalSpec:
    """
    Specification for an eval.
    """

    cls: str
    args: Optional[Dict[str, Any]] = None
    key: Optional[str] = None
    group: Optional[str] = None


@dataclass
class EvalSetSpec:
    """
    Specification for an eval set.
    """

    evals: Sequence[str]
    key: Optional[str] = None
    group: Optional[str] = None


@dataclass
class ModelSpecs:
    completions_: Optional[Sequence[ModelSpec]] = None
    embedding_: Optional[ModelSpec] = None
    ranking_: Optional[ModelSpec] = None

    @property
    def embedding(self) -> ModelSpec:
        if self.embedding_ is None:
            raise ValueError("Embedding model was not specified")
        return self.embedding_

    @property
    def ranking(self) -> ModelSpec:
        if self.ranking_ is None:
            raise ValueError("Ranking model was not specified")
        return self.ranking_

    @property
    def completion(self) -> ModelSpec:
        if self.completions_ is None:
            raise ValueError("Completion model was not specified")
        return self.completions_[0]

    @property
    def completions(self) -> Sequence[ModelSpec]:
        if self.completions_ is None:
            raise ValueError("Completion model was not specified")
        return self.completions_

    @property
    def names(self) -> dict[str, Sequence[str]]:
        dict = {}
        if self.completions_ is not None:
            dict["completions"] = [model.name for model in self.completions_]
        if self.embedding_ is not None:
            dict["embedding"] = [self.embedding_.name]
        if self.ranking_ is not None:
            dict["ranking"] = [self.ranking_.name]
        return dict


@dataclass
class RunSpec:
    model_name: str
    model_names: dict[str, Sequence[str]]
    eval_name: str
    base_eval: str
    split: str
    run_config: Dict[str, Any]
    created_by: str
    run_id: str = None
    created_at: str = None

    def __post_init__(self):
        now = datetime.datetime.utcnow()
        rand_suffix = base64.b32encode(os.urandom(5)).decode("ascii")
        self.run_id = now.strftime("%y%m%d%H%M%S") + rand_suffix
        self.created_at = str(now)
