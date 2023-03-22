from typing import Optional, Text

from evals.plugin.base import Plugin, api_description, namespace


@namespace(
    "MyBirds",
    description="Use the birds plugin to get a list of all the birds you have added to your birds list.",
)
class MyBirds(Plugin):
    def __init__(self, *args, **kwargs) -> None:
        self.birds = ["Red-tailed hawk", "Mountain chickadee", "Black-billed magpie"]
        super().__init__()

    @api_description(name="listBirds", description="API for fetching your birds.")
    def list_birds(cls) -> dict:
        return {"birds": ["Red-tailed hawk", "Mountain chickadee", "Black-billed magpie"]}

    @api_description(
        name="hasBird",
        description="Determine if a bird is in your list",
        api_args={
            "name": {
                "type": "string",
                "optional": False,
                "description": "The name of the bird you want to fetch.",
            }
        },
    )
    def has_bird(self, name: Text) -> dict:
        return {"answer": name in self.birds}
