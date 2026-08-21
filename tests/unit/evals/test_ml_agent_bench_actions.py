from typing import Any


def _actions(monkeypatch: Any):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    from evals.elsuite.hr_ml_agent_bench import actions

    return actions


def test_missing_action_input_is_invalid(monkeypatch: Any) -> None:
    actions = _actions(monkeypatch)
    valid_action = actions.ACTION_SPACE[0]

    parsed = actions.get_action(f"Action: {valid_action.name}")

    assert parsed is not None
    assert parsed.args is None
    assert not actions.is_valid_action(parsed)


def test_valid_action_arguments_are_unchanged(monkeypatch: Any) -> None:
    actions = _actions(monkeypatch)
    valid_action = actions.ACTION_SPACE[0]
    args = {key: None for key in valid_action.usage}
    action = actions.Action(name=valid_action.name, args=args)

    assert actions.is_valid_action(action)
