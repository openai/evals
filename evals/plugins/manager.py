from importlib.metadata import EntryPoint

from stevedore import _cache

from evals.base import ModelSpec

from .base import _ModelRunner


def load_runner(model_spec: ModelSpec) -> _ModelRunner:
    for entry_point in _cache.get_group_all("openevals"):
        if entry_point.name == model_spec.runner:
            runner_class: type[_ModelRunner] = entry_point.load()
            return runner_class()
    raise ValueError(f"Runner {model_spec.runner} not found")


def get_model_spec(name: str) -> ModelSpec:
    # FIXME: use yaml or other configuable way to setup model's spec

    # NOTE: load all plugins
    entry_point: EntryPoint
    for entry_point in _cache.get_group_all("openevals"):
        print(f"entry_point: {entry_point}")
        runner_class: type[_ModelRunner] = entry_point.load()
        resolver = runner_class()

        try:
            model_spec = resolver.resolve(name)
            return model_spec
        except ValueError:
            pass

    raise ValueError(f"Model {name} not found")
