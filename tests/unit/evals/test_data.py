import dataclasses
import json
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import BaseModel

from evals.data import jsondumps


@pytest.mark.parametrize(
    "excluded", [["secret"], ["secret", "secret", "missing"], ["missing"], [], ()]
)
def test_jsondumps_exclusions_do_not_mutate_input(excluded):
    payload = {"keep": 1, "secret": 2, "nested": {"value": 3}}
    original = deepcopy(payload)
    nested = payload["nested"]
    expected = {key: value for key, value in payload.items() if key not in excluded}
    assert json.loads(jsondumps(payload, exclude_keys=excluded)) == expected
    assert json.loads(jsondumps(payload, exclude_keys=excluded)) == expected
    assert payload == original
    assert payload["nested"] is nested


def test_explicit_none_exclusions_match_default_encoding():
    payload = {"keep": 1, "nested": {"value": 3}}
    assert jsondumps(payload, exclude_keys=None) == jsondumps(payload)
    assert payload == {"keep": 1, "nested": {"value": 3}}


@pytest.mark.parametrize("ensure_ascii", [False, True])
def test_standard_json_options_are_preserved(ensure_ascii):
    payload = {"z": "\u0141\u00f3d\u017a", "secret": "omit", "a": [1, 2]}
    expected = {"z": "\u0141\u00f3d\u017a", "a": [1, 2]}
    kwargs = {"ensure_ascii": ensure_ascii, "sort_keys": True, "indent": 2}
    assert jsondumps(payload, exclude_keys=["secret"], **kwargs) == json.dumps(expected, **kwargs)
    assert payload["secret"] == "omit"


@dataclasses.dataclass
class DataRecord:
    value: str
    secret: str


class ModelRecord(BaseModel):
    value: str
    secret: str


@pytest.mark.parametrize("record_type", [DataRecord, ModelRecord])
def test_enhanced_encoder_still_handles_records_and_paths(record_type):
    record = record_type(value="keep", secret="omit")
    payload = {"record": record, "path": Path("synthetic/file"), "secret": "top-level"}
    assert json.loads(jsondumps(payload, exclude_keys=["secret"])) == {
        "record": {"value": "keep"},
        "path": "synthetic/file",
    }
    assert payload["record"] is record
    assert record.secret == "omit"
    assert payload["secret"] == "top-level"


def test_plain_nested_dictionary_exclusions_remain_top_level_only():
    payload = {"secret": "outer", "nested": {"secret": "inner"}}
    assert json.loads(jsondumps(payload, exclude_keys=["secret"])) == {
        "nested": {"secret": "inner"}
    }
    assert payload == {"secret": "outer", "nested": {"secret": "inner"}}


def test_circular_reference_failure_does_not_mutate_input():
    payload = {"secret": "keep-in-memory"}
    payload["cycle"] = payload
    with pytest.raises(ValueError, match="Circular reference"):
        jsondumps(payload, exclude_keys=["secret"])
    assert payload["cycle"] is payload
    assert payload["secret"] == "keep-in-memory"


@pytest.mark.parametrize("payload", [None, [], [1, "text"], "text", 3])
def test_non_dictionary_inputs_keep_existing_behavior(payload):
    assert jsondumps(payload, exclude_keys=["missing"]) == json.dumps(payload, ensure_ascii=False)
