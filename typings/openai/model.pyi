from typing import Optional

from .response import ListResponse

class Model:
    @classmethod
    def list(
        cls,
        api_key: Optional[str] = ...,
        request_id: Optional[str] = ...,
        api_version: Optional[str] = ...,
        organization: Optional[str] = ...,
        api_base: Optional[str] = ...,
        api_type: Optional[str] = ...,
    ) -> ListResponse: ...
