import pytest
import yaml

from evals.base import CompletionFnSpec
from evals.registry import Registry, is_chat_model, n_ctx_from_model_name


def test_n_ctx_from_model_name():
    assert n_ctx_from_model_name("gpt-3.5-turbo") == 4096
    assert n_ctx_from_model_name("gpt-3.5-turbo-0613") == 4096
    assert n_ctx_from_model_name("gpt-3.5-turbo-16k") == 16384
    assert n_ctx_from_model_name("gpt-3.5-turbo-16k-0613") == 16384
    assert n_ctx_from_model_name("gpt-4") == 8192
    assert n_ctx_from_model_name("gpt-4-0613") == 8192
    assert n_ctx_from_model_name("gpt-4-32k") == 32768
    assert n_ctx_from_model_name("gpt-4-32k-0613") == 32768
    assert n_ctx_from_model_name("gpt-3.5-turbo") == 4096
    assert n_ctx_from_model_name("gpt-3.5-turbo-0314") == 4096
    assert n_ctx_from_model_name("gpt-3.5-turbo-0613") == 4096
    assert n_ctx_from_model_name("gpt-3.5-turbo-16k") == 16384
    assert n_ctx_from_model_name("gpt-3.5-turbo-16k-0314") == 16384
    assert n_ctx_from_model_name("gpt-3.5-turbo-16k-0613") == 16384


def test_is_chat_model():
    assert is_chat_model("gpt-3.5-turbo")
    assert is_chat_model("gpt-3.5-turbo-0613")
    assert is_chat_model("gpt-3.5-turbo-16k")
    assert is_chat_model("gpt-3.5-turbo-16k-0613")
    assert is_chat_model("gpt-4")
    assert is_chat_model("gpt-4-0613")
    assert is_chat_model("gpt-4-32k")
    assert is_chat_model("gpt-4-32k-0613")
    assert not is_chat_model("text-davinci-003")
    assert not is_chat_model("gpt4-base")
    assert not is_chat_model("code-davinci-002")


def test_dereference_resolves_alias_chains():
    registry = Registry([])
    aliases = {
        "short-name": {"id": "middle-name"},
        "middle-name": {"id": "target"},
        "target": {"cls": "example:Target", "args": {"original": True}},
    }

    spec = registry._dereference(
        "short-name",
        aliases,
        "completion_fn",
        CompletionFnSpec,
        args={"override": True},
    )

    assert spec is not None
    assert spec.cls == "example:Target"
    assert spec.args == {"override": True}


@pytest.mark.parametrize(
    "name,aliases",
    [
        ("self", {"self": {"id": "self"}}),
        (
            "first",
            {
                "first": "second",
                "second": {"id": "third"},
                "third": {"id": "first"},
            },
        ),
    ],
)
def test_dereference_rejects_alias_cycles(name, aliases):
    class BoundedAliases(dict):
        lookups = 0

        def __getitem__(self, key):
            self.lookups += 1
            if self.lookups > (4 * len(self)) + 1:
                raise AssertionError("alias resolution did not detect a cycle")
            return super().__getitem__(key)

    with pytest.raises(ValueError, match="Alias cycle detected"):
        Registry([])._dereference(name, BoundedAliases(aliases), "completion_fn", dict)


def test_dereference_preserves_missing_alias_target_error():
    with pytest.raises(KeyError, match="missing"):
        Registry([])._dereference("alias", {"alias": {"id": "missing"}}, "completion_fn", dict)


def test_public_registry_loads_yaml_aliases_for_eval_and_completion_fn(tmp_path):
    evals_path = tmp_path / "evals"
    evals_path.mkdir()
    (evals_path / "examples.yaml").write_text(
        yaml.safe_dump(
            {
                "eval-alias": {"id": "eval-middle"},
                "eval-middle": {"id": "eval-target"},
                "eval-target": {"class": "example:Eval", "args": {}},
            }
        ),
        encoding="utf-8",
    )
    completion_fns_path = tmp_path / "completion_fns"
    completion_fns_path.mkdir()
    (completion_fns_path / "examples.yaml").write_text(
        yaml.safe_dump(
            {
                "completion-alias": {"id": "completion-middle"},
                "completion-middle": {"id": "completion-target"},
                "completion-target": {"class": "example:CompletionFn", "args": {}},
            }
        ),
        encoding="utf-8",
    )

    registry = Registry([tmp_path])

    eval_spec = registry.get_eval("eval-alias")
    completion_fn_spec = registry.get_completion_fn("completion-alias")

    assert eval_spec is not None
    assert eval_spec.cls == "example:Eval"
    assert eval_spec.key == "eval-target"
    assert completion_fn_spec is not None
    assert completion_fn_spec.cls == "example:CompletionFn"
    assert completion_fn_spec.key == "completion-target"
