from datetime import datetime
from typing import Optional, Text

import pytz

from evals.plugin.base import Plugin, api_description, namespace


@namespace(
    "Clock",
    description="A clock plugin that displays time in the specified timezone.",
)
class Clock(Plugin):
    def __init__(self, default_timezone: Optional[Text] = "UTC", *args, **kwargs) -> None:
        self.default_timezone = default_timezone
        super().__init__()

    @api_description(
        name="getTime",
        description="API for fetching the current time in the specified timezone.",
        api_args={
            "timezone": {
                "type": "string",
                "optional": True,
                "description": "The timezone you want to fetch the time for (e.g., 'America/Los_Angeles').",
            }
        },
    )
    def get_time(self, timezone: Optional[Text] = None) -> dict:
        if timezone is None:
            timezone = self.default_timezone

        try:
            tz = pytz.timezone(timezone)
        except pytz.UnknownTimeZoneError:
            return {"error": f"Unknown timezone: {timezone}"}

        current_time = datetime.now(tz)
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        return {"time": formatted_time, "timezone": timezone}
