from .response import ListResponse

class Model:
    @classmethod
    def list(
        cls,
        api_key: str | None = ...,
        request_id: str | None = ...,
        api_version: str | None = ...,
        organization: str | None = ...,
        api_base: str | None = ...,
        api_type: str | None = ...,
    ) -> ListResponse: ...
