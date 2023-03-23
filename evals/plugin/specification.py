from typing import Any, Optional

from pydantic import BaseModel


def comment_content(content: str) -> str:
    return "".join(
        [
            f"// {clean_line}\n"
            for line in content.strip().split("\n")
            if (clean_line := line.strip())
        ]
    )


class Param(BaseModel):
    name: str
    type: str
    required: bool
    default: Any
    description: Optional[str] = None

    def to_typescript(self):
        colon = ":" if self.required else "?:"
        typescript = ""
        if self.description:
            typescript += comment_content(self.description)
        typescript += f"{self.name}{colon} {self.type},"
        if self.default is not None:
            typescript += f" // default: {self.default}"
        return typescript


class KwargsFunctionSignature(BaseModel):
    name: str
    description: Optional[str] = None
    params: list[Param]

    def get_param(self, name: str) -> Param:
        for param in self.params:
            if param.name == name:
                return param
        raise KeyError(f"Param {name} not found in {self.name}.")

    def to_typescript(self):
        comment = ""

        if self.description:
            comment += comment_content(self.description)

        if not self.params:
            return comment + f"type {self.name} = () => any;"

        return comment + "\n".join(
            [
                f"type {self.name} = (_: {{",
                *[param.to_typescript() for param in self.params],
                "}) => any;",
            ]
        )


class Api(BaseModel):
    """An API that can be called by the model."""

    namespace: str
    description_for_model: str
    signatures: list[KwargsFunctionSignature]

    def to_typescript(self):
        return "\n\n".join(
            [
                f"namespace {self.namespace} {{",
                *(s.to_typescript() for s in self.signatures),
                f"}} // namespace {self.namespace}",
            ]
        )
