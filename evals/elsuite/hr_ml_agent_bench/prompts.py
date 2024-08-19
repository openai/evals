import json
import textwrap

from evals.elsuite.hr_ml_agent_bench.actions import ACTION_SPACE
from evals.elsuite.hr_ml_agent_bench.schema import ActionInfo

ACTION_FORMAT = """
- {name}:
    {description}
    
    Usage:
        ```
        Action: {name}
        Action Input: {usage}
        ```
    
    Returns:
        {return_value}
""".strip()

USAGE_INSTRUCTIONS = """
Always respond in this format exactly:

```
Action: The action to take, which should be one of the names of the actions above.
Action Input: The input to the action as a valid JSON string.
```

For example, to `List Files` in the current directory, you would write:

```
Action: List Files
Action Input: {
    "dir_path": "."
}
```
""".strip()


def format_action(action: ActionInfo) -> str:
    """Formats a single action into a descriptive string."""

    usage = json.dumps(action.usage, indent=4, ensure_ascii=False)
    indented_usage = textwrap.indent(text=usage, prefix=" " * 8)
    indented_usage = indented_usage.lstrip()

    return ACTION_FORMAT.format(
        name=action.name,
        description=action.description,
        usage=indented_usage,
        return_value=action.return_value,
    )


def get_actions_description(actions: list[ActionInfo]) -> str:
    """Formats a list of actions into a descriptive string."""

    return "\n\n".join(format_action(action) for action in actions)


def get_task_description(research_problem: str) -> str:
    """Get a description of the task and available actions."""

    prompt = "You have access to the following actions:\n\n"
    prompt += get_actions_description(ACTION_SPACE)
    prompt += f"\n\nResearch Problem: {research_problem}\n\n"
    prompt += USAGE_INSTRUCTIONS

    return prompt
