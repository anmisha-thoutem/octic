from datetime import datetime
from zoneinfo import ZoneInfo

from config.settings import settings


def get_current_datetime() -> dict:
    timezone_name = settings.APP_TIMEZONE or "UTC"
    timezone = ZoneInfo(timezone_name)
    current = datetime.now(timezone)
    return {
        "timezone": timezone_name,
        "datetime": current.isoformat(),
        "date": current.date().isoformat(),
        "time": current.strftime("%H:%M:%S %Z"),
    }
