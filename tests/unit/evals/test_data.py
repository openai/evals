import json

from evals.data import jsondumps


def test_jsondumps_exclude_keys_does_not_mutate_input() -> None:
    payload = {"keep": 1, "secret": 2}

    encoded = jsondumps(payload, exclude_keys=["secret"])

    assert json.loads(encoded) == {"keep": 1}
    assert payload == {"keep": 1, "secret": 2}


def test_jsondumps_ignores_missing_excluded_keys() -> None:
    payload = {"keep": 1}

    encoded = jsondumps(payload, exclude_keys=["missing"])

    assert json.loads(encoded) == payload
    assert payload == {"keep": 1}
