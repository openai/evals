from typing import List, Text

from evals.plugin.base import Plugin, api_description, namespace


@namespace(
    "TodoList",
    description="A simple todo list plugin.",
)
class TodoList(Plugin):
    def __init__(self, *args, **kwargs) -> None:
        self.todos: List[Text] = []
        super().__init__()

    @api_description(name="listItems", description="API for fetching all TODO items.")
    def list_todos(self) -> dict:
        return {
            "items": self.todos,
        }

    @api_description(
        name="addTodo",
        description="Add a new TODO item",
        api_args={
            "item": {
                "type": "string",
                "optional": False,
                "description": "Description of the todo item.",
            }
        },
    )
    def add_todo(self, item: Text) -> dict:
        self.todos.append(item)
        return {}

    @api_description(
        name="removeTodo",
        description="Remove a TODO item",
        api_args={
            "item": {
                "type": "string",
                "optional": False,
                "description": "Description of the todo item.",
            }
        },
    )
    def remove_todo(self, item: Text) -> dict:
        self.todos.remove(item)
        return {}
