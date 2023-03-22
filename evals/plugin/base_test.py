import json

from evals.plugin.base import Plugin
from evals.plugin.openai.bird import MyBirds
from evals.plugin.openai.clock import Clock


def test_description() -> None:
    plugin = MyBirds()
    desc = plugin.description()
    assert (
        desc
        == """// Use the birds plugin to get a list of all the birds you have added to your birds list.
namespace MyBirds {

// Determine if a bird is in your list
type hasBird = (_: {
// The name of the bird you want to fetch.
name: string,
}) => any;

// API for fetching your birds.
type listBirds = () => any;

} // namespace MyBirds"""
    )


def test_invoke_tool_with_args() -> None:
    plugin = MyBirds()
    result = Plugin.invoke(plugin.has_bird, {"name": "Red-tailed hawk"})
    assert "content" in result
    data = json.loads(result["content"])
    assert data == {"answer": True}
    assert data == plugin.has_bird(name="Red-tailed hawk")


def test_invoke_tool_with_no_args() -> None:
    plugin = MyBirds()
    result = Plugin.invoke(plugin.list_birds)
    assert "content" in result
    data = json.loads(result["content"])
    assert data == {"birds": plugin.birds}
    assert data == plugin.list_birds()
