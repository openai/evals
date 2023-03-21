"""
Functions to handle registration of evals. To add a new eval to the registry,
add an entry in one of the YAML files in the `../registry` dir.
By convention, every eval name should start with {base_eval}.{split}.
"""

import functools
import logging
import os
import re
from functools import partial
from pathlib import Path
from typing import Any, Iterator, Sequence, Type, Union

import yaml

from evals.base import BaseEvalSpec, EvalSetSpec, EvalSpec
from evals.utils.misc import make_object

logger = logging.getLogger(__name__)

DEFAULT_PATHS = [Path(__file__).parents[0].resolve() / "registry", Path.home() / ".evals"]


class Registry:
    def __init__(self, registry_paths: Sequence[Union[str, Path]] = DEFAULT_PATHS):
        self._registry_paths = [Path(p) if isinstance(p, str) else p for p in registry_paths]

    def make_callable(self, spec):
        return partial(make_object(spec.cls).create_and_run, **(spec.args or {}))

    def get_class(self, spec: dict) -> Any:
        return make_object(spec.cls, **(spec.args if spec.args else {}))

    def _dereference(self, name: str, d: dict, object: str, type: Type) -> dict:
        if not name in d:
            return None

        def get_alias():
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

        try:
            return type(**spec)
        except TypeError as e:
            raise TypeError(f"Error while processing {object} {name}: {e}")

    def get_eval(self, name: str) -> EvalSpec:
        return self._dereference(name, self._evals, "eval", EvalSpec)

    def get_eval_set(self, name: str) -> EvalSetSpec:
        return self._dereference(name, self._eval_sets, "eval set", EvalSetSpec)

    def get_evals(self, patterns: Sequence[str]) -> Iterator[EvalSpec]:
        # valid patterns: hello, hello.dev*, hello.dev.*-v1
        def get_regexp(pattern):
            pattern = pattern.replace(".", "\\.")
            pattern = pattern.replace("*", ".*")
            return re.compile(f"^{pattern}$")

        regexps = list(map(get_regexp, patterns))
        for name in self._evals:
            # if any regexps match, return the name
            if any(map(lambda regexp: regexp.match(name), regexps)):
                yield self.get_eval(name)

    def get_base_evals(self) -> list[BaseEvalSpec]:
        base_evals = []
        for name, spec in self._evals.items():
            if name.count(".") == 0:
                base_evals.append(self.get_base_eval(name))
        return base_evals

    def get_base_eval(self, name: str) -> BaseEvalSpec:
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

    def _process_file(self, registry, path):
        with open(path, "r") as f:
            d = yaml.safe_load(f)

        if d is None:
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

    def _process_directory(self, registry, path):
        files = Path(path).glob("*.yaml")
        for file in files:
            self._process_file(registry, file)

    def _load_registry(self, paths):
        registry = {}
        for path in paths:
            logging.info(f"Loading registry from {path}")
            if os.path.exists(path):
                if os.path.isdir(path):
                    self._process_directory(registry, path)
                else:
                    self._process_file(registry, path)
        return registry

    @functools.cached_property
    def _eval_sets(self):
        return self._load_registry([p / "eval_sets" for p in self._registry_paths])

    @functools.cached_property
    def _evals(self):
        return self._load_registry([p / "evals" for p in self._registry_paths])


registry = Registry()
