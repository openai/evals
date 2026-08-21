from evals.elsuite.basic.json_match import _MISSING, json_match


def test_json_null_matches_json_null() -> None:
    assert json_match(None, None)
    assert json_match({"value": None}, {"value": None})
    assert json_match([1, None, 3], [1, None, 3])


def test_missing_key_does_not_match_json_null() -> None:
    assert not json_match({}, {"value": None})
    assert not json_match({"value": None}, {})


def test_missing_sentinel_never_matches_json_null() -> None:
    assert not json_match(_MISSING, None)
    assert not json_match(None, _MISSING)
