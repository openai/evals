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
from typing import Any, Generator, Iterator, Optional, Sequence, Tuple, Type, TypeVar, Union

import openai
import yaml
from openai import OpenAI

from evals import OpenAIChatCompletionFn, OpenAICompletionFn
from evals.api import CompletionFn, DummyCompletionFn
from evals.base import BaseEvalSpec, CompletionFnSpec, EvalSetSpec, EvalSpec
from evals.elsuite.modelgraded.base import ModelGradedSpec
from evals.utils.misc import make_object

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

logger = logging.getLogger(__name__)

DEFAULT_PATHS = [
    Path(__file__).parents[0].resolve() / "registry",
    Path.home() / ".evals",
]
SPEC_RESERVED_KEYWORDS = ["key", "group", "cls"]


def n_ctx_from_model_name(model_name: str) -> Optional[int]:
    """Returns n_ctx for a given API model name. Model list last updated 2023-10-24."""
    # note that for most models, the max tokens is n_ctx + 1
    PREFIX_AND_N_CTX: list[tuple[str, int]] = [
        ("gpt-3.5-turbo-16k-", 16384),
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
        "gpt-3.5-turbo-16k": 16384,
        "gpt-3.5-turbo-instruct": 4096,
        "gpt-3.5-turbo-instruct-0914": 4096,
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
        "gpt-4-base": 8192,
        "gpt-4-1106-preview": 128_000,
        "gpt-4-turbo-preview": 128_000,
        "gpt-4-0125-preview": 128_000,
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
    if model_name in {"gpt-4-base"} or model_name.startswith("gpt-3.5-turbo-instruct"):
        return False

    CHAT_MODEL_NAMES = {"gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k", "gpt-4o"}

    if model_name in CHAT_MODEL_NAMES:
        return True

    for model_prefix in {"gpt-3.5-turbo-", "gpt-4-"}:
        if model_name.startswith(model_prefix):
            return True

    return False


T = TypeVar("T")
RawRegistry = dict[str, Any]


class Registry:
    def __init__(self, registry_paths: Sequence[Union[str, Path]] = DEFAULT_PATHS):
        self._registry_paths = [Path(p) if isinstance(p, str) else p for p in registry_paths]

    def add_registry_paths(self, paths: Sequence[Union[str, Path]]) -> None:
        self._registry_paths.extend([Path(p) if isinstance(p, str) else p for p in paths])

    @cached_property
    def api_model_ids(self) -> list[str]:
        try:
            return [m.id for m in client.models.list().data]
        except openai.OpenAIError as err:
            # Errors can happen when running eval with completion function that uses custom
            # API endpoints and authentication mechanisms.
            logger.warning(f"Could not fetch API model IDs from OpenAI API: {err}")
            return []

    def make_completion_fn(
        self,
        name: str,
        **kwargs: Any,
    ) -> CompletionFn:
        """
        Create a CompletionFn. The name can be one of the following formats:
        1. openai-model-id (e.g. "gpt-3.5-turbo")
        2. completion-fn-id (from the registry)
        """
        if name == "dummy":
            return DummyCompletionFn()

        n_ctx = n_ctx_from_model_name(name)

        if is_chat_model(name):
            return OpenAIChatCompletionFn(model=name, n_ctx=n_ctx, **kwargs)
        elif name in self.api_model_ids:
            return OpenAICompletionFn(model=name, n_ctx=n_ctx, **kwargs)

        # No match, so try to find a completion-fn-id in the registry
        spec = self.get_completion_fn(name) or self.get_solver(name)
        if spec is None:
            raise ValueError(f"Could not find CompletionFn/Solver in the registry with ID {name}")
        if spec.args is None:
            spec.args = {}
        spec.args.update(kwargs)

        spec.args["registry"] = self
        instance = make_object(spec.cls)(**spec.args or {})
        assert isinstance(instance, CompletionFn), f"{name} must be a CompletionFn"
        return instance

    def get_class(self, spec: EvalSpec) -> Any:
        return make_object(spec.cls, **(spec.args if spec.args else {}))

    def _dereference(
        self, name: str, d: RawRegistry, object: str, type: Type[T], **kwargs: dict
    ) -> Optional[T]:
        if name not in d:
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
        return self._dereference(
            name, self._completion_fns | self._solvers, "completion_fn", CompletionFnSpec
        )

    def get_solver(self, name: str) -> Optional[CompletionFnSpec]:
        return self._dereference(name, self._solvers, "solver", CompletionFnSpec)

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
        if name not in self._evals:
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

    def _load_file(self, path: Path) -> Generator[Tuple[str, Path, dict], None, None]:
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)

        if d is None or not isinstance(d, dict):
            # no entries in the file
            return

        for name, spec in d.items():
            yield name, path, spec

    def _load_directory(self, path: Path) -> Generator[Tuple[str, Path, dict], None, None]:
        files = Path(path).glob("*.yaml")
        for file in files:
            yield from self._load_file(file)

    def _load_resources(
        self, registry_path: Path, resource_type: str
    ) -> Generator[Tuple[str, Path, dict], None, None]:
        path = registry_path / resource_type
        logging.info(f"Loading registry from {path}")

        if os.path.exists(path):
            if os.path.isdir(path):
                yield from self._load_directory(path)
            else:
                yield from self._load_file(path)

    @staticmethod
    def _validate_reserved_keywords(spec: dict, name: str, path: Path) -> None:
        for reserved_keyword in SPEC_RESERVED_KEYWORDS:
            if reserved_keyword in spec:
                raise ValueError(
                    f"{reserved_keyword} is a reserved keyword, but was used in {name} from {path}"
                )

    def _load_registry(self, registry_paths: Sequence[Path], resource_type: str) -> RawRegistry:
        """Load registry from a list of regstry paths and a specific resource type

        Each path includes yaml files which are a dictionary of name -> spec.
        """

        registry: RawRegistry = {}

        for registry_path in registry_paths:
            for name, path, spec in self._load_resources(registry_path, resource_type):
                assert name not in registry, f"duplicate entry: {name} from {path}"
                self._validate_reserved_keywords(spec, name, path)

                spec["key"] = name
                spec["group"] = str(os.path.basename(path).split(".")[0])
                spec["registry_path"] = registry_path

                if "class" in spec:
                    spec["cls"] = spec["class"]
                    del spec["class"]

                registry[name] = spec

        return registry

    @functools.cached_property
    def _completion_fns(self) -> RawRegistry:
        return self._load_registry(self._registry_paths, "completion_fns")

    @functools.cached_property
    def _solvers(self) -> RawRegistry:
        return self._load_registry(self._registry_paths, "solvers")

    @functools.cached_property
    def _eval_sets(self) -> RawRegistry:
        return self._load_registry(self._registry_paths, "eval_sets")

    @functools.cached_property
    def _evals(self) -> RawRegistry:
        return self._load_registry(self._registry_paths, "evals")

    @functools.cached_property
    def _modelgraded_specs(self) -> RawRegistry:
        return self._load_registry(self._registry_paths, "modelgraded")


registry = Registry()
