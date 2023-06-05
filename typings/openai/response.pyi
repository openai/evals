from typing import Any, Literal, TypedDict

class ListResponse(TypedDict):
    """Response from Model.list

    Reference: https://platform.openai.com/docs/api-reference/models"""

    object: Literal["list"]
    data: list[Model]

class Model(TypedDict):
    id: str
    object: Literal["model"]
    owned_by: str
    permission: list[Any]  # TODO
