import re
from functools import lru_cache


TIMESTAMP_PATTERN = re.compile(
    r"^(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})(?:\.(?P<millis>\d{1,3}))?$"
)


@lru_cache(maxsize=1024)
def timestamp_to_seconds(value: str) -> float:
    """
    Converts a HH:MM:SS.mmm (or HH:MM:SS) style timestamp into seconds.
    """
    match = TIMESTAMP_PATTERN.match(value.strip())
    if not match:
        raise ValueError(f"Unsupported timestamp format: {value}")
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    second = int(match.group("second"))
    millis = match.group("millis")
    millis_val = int(millis) if millis else 0
    return hour * 3600 + minute * 60 + second + millis_val / 1000.0


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))
