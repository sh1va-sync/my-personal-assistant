from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.config.settings import get_settings


def get_timezone(name: str | None = None) -> ZoneInfo:
    tz_name = (name or get_settings().default_timezone).strip()
    try:
        return ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        return ZoneInfo(get_settings().default_timezone)


def now_tz(timezone_name: str | None = None) -> datetime:
    return datetime.now(get_timezone(timezone_name))
