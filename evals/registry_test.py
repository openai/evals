import pytest
from mock import ANY, patch

from evals.api import DummyCompletionFn
from evals.elsuite.modelgraded.classify_utils import get_choice_strings
from evals.record import DummyRecorder
from evals.registry import Registry, is_chat_model, n_ctx_from_model_name
from evals.utils.test import TestCompletionFn

LFS_POINTER_PREFIX = "version https://git-lfs.github.com/spec/v1"

registry = Registry()


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


def _make_eval(spec, completion_fn, **args):
    """Instantiate an eval from its registry spec the way `oaieval` does."""
    eval_cls = registry.get_class(spec)
    return eval_cls(
        completion_fns=[completion_fn],
        eval_registry_path=spec.registry_path,
        name=spec.key,
        registry=registry,
        **args,
    )


def _skip_if_lfs_pointer(path):
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            if f.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX:
                pytest.skip(f"{path} is a Git LFS pointer; run `git lfs pull`")


def _metaeval_specs():
    # Aliases resolve to the same spec; keying by spec.key yields each metaeval once.
    specs_by_key = {spec.key: spec for spec in registry.get_evals(["*"])}
    return [
        pytest.param(spec, id=key)
        for key, spec in sorted(specs_by_key.items())
        if (spec.args or {}).get("metaeval")
    ]


@pytest.mark.parametrize("spec", _metaeval_specs())
def test_metaeval_labels_are_valid_choices(spec):
    """
    ModelBasedClassify scores a metaeval as `choice == sample["choice"]`, where `choice` is one of
    the modelgraded spec's choice_strings, so a label outside that set can never be matched.
    """
    metaeval = _make_eval(spec, DummyCompletionFn())
    _skip_if_lfs_pointer(metaeval._get_samples_path())

    choices = set(get_choice_strings(metaeval.mg.choice_strings))
    labels = {sample["choice"] for sample in metaeval.get_samples()}
    assert labels <= choices, f"labels {labels - choices} are not among choices {choices}"


@pytest.mark.parametrize("label", ["Yes", "No"])
def test_naughty_strings_graded_scores_the_labeled_answer_as_correct(label):
    """
    naughty_strings_graded samples already carry completions, so it can only measure the grader:
    a grader that gives a sample's labeled answer must get a metascore of True.
    """
    spec = registry.get_eval("naughty_strings_graded")
    naughty_strings_graded = _make_eval(spec, TestCompletionFn(label))
    _skip_if_lfs_pointer(naughty_strings_graded._get_samples_path())
    sample = next(s for s in naughty_strings_graded.get_samples() if s["choice"] == label)

    recorder = DummyRecorder(None)
    with recorder.as_default_recorder("x"), patch.object(
        recorder, "record_metrics", wraps=recorder.record_metrics
    ) as record_metrics:
        naughty_strings_graded.eval_sample(sample, None)
        record_metrics.assert_called_once_with(choice=label, score=ANY, metascore=True)
