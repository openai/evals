import dataclasses
from typing import Optional, Text

from pydantic import BaseModel

from evals.data import jsondumps


class MyPydanticClass(BaseModel):
    first_name: Text
    last_name: Text


@dataclasses.dataclass
class MyDataClass:
    first_name: Text
    last_name: Text
    sub_class: Optional[MyPydanticClass] = None


def test_jsondumps():
    assert '{"first_name": "a", "last_name": "b", "sub_class": null}' == jsondumps(
        MyDataClass(first_name="a", last_name="b")
    )
    assert '{"first_name": "a", "sub_class": null}' == jsondumps(
        MyDataClass(first_name="a", last_name="b"), exclude_keys=["last_name"]
    )
    assert '{"first_name": "a", "last_name": "b"}' == jsondumps(
        MyPydanticClass(first_name="a", last_name="b")
    )
    assert '{"first_name": "a"}' == jsondumps(
        MyPydanticClass(first_name="a", last_name="b"), exclude_keys=["last_name"]
    )
    assert '{"first_name": "a", "last_name": "b"}' == jsondumps(
        {"first_name": "a", "last_name": "b"}
    )
    assert '{"first_name": "a"}' == jsondumps(
        {"first_name": "a", "last_name": "b"}, exclude_keys=["last_name"]
    )
    assert '{"first_name": "a", "sub_class": {"first_name": "a"}}' == jsondumps(
        MyDataClass("a", "b", MyPydanticClass(first_name="a", last_name="b")),
        exclude_keys=["last_name"],
    )
