"""
Functions to handle registration of evals. To add a new eval to the registry,
add an entry in one of the YAML files in the `../registry` dir.
By convention, every eval name should start with {base_eval}.{split}.
"""

import copy
import difflib
import functools
import logging
import os
import re
from functools import cached_property
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence, Type, TypeVar, Union

import openai
import yaml

from evals import OpenAIChatCompletionFn, OpenAICompletionFn
from evals.api import CompletionFn, DummyCompletionFn
from evals.base import BaseEvalSpec, CompletionFnSpec, EvalSetSpec, EvalSpec
from evals.elsuite.modelgraded.base import ModelGradedSpec
from evals.utils.misc import make_object

logger = logging.getLogger(__name__)

DEFAULT_PATHS = [Path(__file__).parents[0].resolve() / "registry", Path.home() / ".evals"]


def n_ctx_from_model_name(model_name: str) -> Optional[int]:
    """Returns n_ctx for a given API model name. Model list last updated 2023-06-16."""
    # note that for most models, the max tokens is n_ctx + 1
    PREFIX_AND_N_CTX: list[tuple[str, int]] = [
        ("gpt-3.5-turbo-", 4096),
        ("gpt-4-32k-", 32768),
        ("gpt-4-", 8192),
    ]
    MODEL_NAME_TO_N_CTX: dict[str, int] = {
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
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
        "gpt-4-base": 8192,
    }

    # first, look for an exact match
    if model_name in MODEL_NAME_TO_N_CTX:
        return MODEL_NAME_TO_N_CTX[model_name]

    # otherwise, look for a prefix match
    for model_prefix, n_ctx in PREFIX_AND_N_CTX:
        if model_name.startswith(model_prefix):
            return n_ctx

    # not found
    return None


def is_chat_model(model_name: str) -> bool:
    if model_name in {"gpt-4-base"}:
        return False

    CHAT_MODEL_NAMES = {"gpt-3.5-turbo", "gpt-4", "gpt-4-32k"}
    if model_name in CHAT_MODEL_NAMES:
        return True

    for model_prefix in {"gpt-3.5-turbo-", "gpt-4-", "gpt-4-32k-"}:
        if model_name.startswith(model_prefix):
            return True
    return False


T = TypeVar("T")
RawRegistry = dict[str, Any]


class Registry:
    def __init__(self, registry_paths: Sequence[Union[str, Path]] = DEFAULT_PATHS):
        self._registry_paths = [Path(p) if isinstance(p, str) else p for p in registry_paths]

    def add_registry_paths(self, paths: list[Union[str, Path]]) -> None:
        self._registry_paths.extend([Path(p) if isinstance(p, str) else p for p in paths])

    @cached_property
    def api_model_ids(self) -> list[str]:
        try:
            return [m["id"] for m in openai.Model.list()["data"]]
        except openai.error.OpenAIError as err:  # type: ignore
            # Errors can happen when running eval with completion function that uses custom
            # API endpoints and authentication mechanisms.
            logger.warning(f"Could not fetch API model IDs from OpenAI API: {err}")
            return []

    def make_completion_fn(self, name: str) -> CompletionFn:
        """
        Create a CompletionFn. The name can be one of the following formats:
        1. openai-model-id (e.g. "gpt-3.5-turbo")
        2. completion-fn-id (from the registry)
        """

        if name == "dummy":
            return DummyCompletionFn()

        n_ctx = n_ctx_from_model_name(name)

        if is_chat_model(name):
            return OpenAIChatCompletionFn(model=name, n_ctx=n_ctx)
        elif name in self.api_model_ids:
            return OpenAICompletionFn(model=name, n_ctx=n_ctx)

        # No match, so try to find a completion-fn-id in the registry
        spec = self.get_completion_fn(name)
        if spec is None:
            raise ValueError(f"Could not find CompletionFn in the registry with ID {name}")
        if spec.args is None:
            spec.args = {}

        spec.args["registry"] = self
        instance = make_object(spec.cls)(**spec.args or {})
        assert isinstance(instance, CompletionFn), f"{name} must be a CompletionFn"
        return instance

    def get_class(self, spec: EvalSpec) -> Any:
        return make_object(spec.cls, **(spec.args if spec.args else {}))

    def _dereference(
        self, name: str, d: RawRegistry, object: str, type: Type[T], **kwargs: dict
    ) -> Optional[T]:
        if not name in d:
            logger.warning(
                (
                    f"{object} '{name}' not found. "
                    f"Closest matches: {difflib.get_close_matches(name, d.keys(), n=5)}"
                )
            )
            return None

        def get_alias() -> Optional[str]:
            if isinstance(d[name], str):
                return d[name]
            if isinstance(d[name], dict) and "id" in d[name]:
                return d[name]["id"]
            return None

        logger.debug(f"Looking for {name}")
        while True:
            alias = get_alias()

            if alias is None:
                break
            name = alias

        spec = d[name]
        if kwargs:
            spec = copy.deepcopy(spec)
            spec.update(kwargs)

        try:
            return type(**spec)
        except TypeError as e:
            raise TypeError(f"Error while processing {object} '{name}': {e}")

    def get_modelgraded_spec(self, name: str, **kwargs: dict) -> Optional[ModelGradedSpec]:
        assert name in self._modelgraded_specs, (
            f"Modelgraded spec {name} not found. "
            f"Closest matches: {difflib.get_close_matches(name, self._modelgraded_specs.keys(), n=5)}"
        )
        return self._dereference(
            name, self._modelgraded_specs, "modelgraded spec", ModelGradedSpec, **kwargs
        )

    def get_completion_fn(self, name: str) -> Optional[CompletionFnSpec]:
        return self._dereference(name, self._completion_fns, "completion_fn", CompletionFnSpec)

    def get_eval(self, name: str) -> Optional[EvalSpec]:
        return self._dereference(name, self._evals, "eval", EvalSpec)

    def get_eval_set(self, name: str) -> Optional[EvalSetSpec]:
        return self._dereference(name, self._eval_sets, "eval set", EvalSetSpec)

    def get_evals(self, patterns: Sequence[str]) -> Iterator[Optional[EvalSpec]]:
        # valid patterns: hello, hello.dev*, hello.dev.*-v1
        def get_regexp(pattern: str) -> re.Pattern[str]:
            pattern = pattern.replace(".", "\\.")
            pattern = pattern.replace("*", ".*")
            return re.compile(f"^{pattern}$")

        regexps = list(map(get_regexp, patterns))
        for name in self._evals:
            # if any regexps match, return the name
            if any(map(lambda regexp: regexp.match(name), regexps)):
                yield self.get_eval(name)

    def get_base_evals(self) -> list[Optional[BaseEvalSpec]]:
        base_evals: list[Optional[BaseEvalSpec]] = []
        for name, spec in self._evals.items():
            if name.count(".") == 0:
                base_evals.append(self.get_base_eval(name))
        return base_evals

    def get_base_eval(self, name: str) -> Optional[BaseEvalSpec]:
        if not name in self._evals:
            return None

        spec_or_alias = self._evals[name]
        if isinstance(spec_or_alias, dict):
            spec = spec_or_alias
            try:
                return BaseEvalSpec(**spec)
            except TypeError as e:
                raise TypeError(f"Error while processing base eval {name}: {e}")

        alias = spec_or_alias
        return BaseEvalSpec(id=alias)

    def _process_file(self, registry: RawRegistry, path: Path) -> None:
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)

        if d is None or not isinstance(d, dict):
            # no entries in the file
            return

        for name, spec in d.items():
            assert name not in registry, f"duplicate entry: {name} from {path}"
            if isinstance(spec, dict):
                if "key" in spec:
                    raise ValueError(
                        f"key is a reserved keyword, but was used in {name} from {path}"
                    )
                if "group" in spec:
                    raise ValueError(
                        f"group is a reserved keyword, but was used in {name} from {path}"
                    )
                if "cls" in spec:
                    raise ValueError(
                        f"cls is a reserved keyword, but was used in {name} from {path}"
                    )

                spec["key"] = name
                spec["group"] = str(os.path.basename(path).split(".")[0])
                if "class" in spec:
                    spec["cls"] = spec["class"]
                    del spec["class"]
            registry[name] = spec

    def _process_directory(self, registry: RawRegistry, path: Path) -> None:
        files = Path(path).glob("*.yaml")
        for file in files:
            self._process_file(registry, file)

    def _load_registry(self, paths: Sequence[Path]) -> RawRegistry:
        """Load registry from a list of paths.

        Each path or yaml specifies a dictionary of name -> spec.
        """
        registry: RawRegistry = {}
        for path in paths:
            logging.info(f"Loading registry from {path}")
            if os.path.exists(path):
                if os.path.isdir(path):
                    self._process_directory(registry, path)
                else:
                    self._process_file(registry, path)
        return registry

    @functools.cached_property
    def _completion_fns(self) -> RawRegistry:
        return self._load_registry([p / "completion_fns" for p in self._registry_paths])

    @functools.cached_property
    def _eval_sets(self) -> RawRegistry:
        return self._load_registry([p / "eval_sets" for p in self._registry_paths])

    @functools.cached_property
    def _evals(self) -> RawRegistry:
        return self._load_registry([p / "evals" for p in self._registry_paths])

    @functools.cached_property
    def _modelgraded_specs(self) -> RawRegistry:
        return self._load_registry([p / "modelgraded" for p in self._registry_paths])


registry = Registry()
