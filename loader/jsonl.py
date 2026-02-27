import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .base import LoaderBase
from .utils import timestamp_to_seconds


def _extract_nested_value(data, key: str):
    parts = key.split(".")
    value = data
    for part in parts:
        if isinstance(value, dict):
            value = value.get(part)
        else:
            return None
    return value


@dataclass
class JsonlSeries:
    times: np.ndarray
    values: Dict[str, np.ndarray] = field(default_factory=dict)


class JsonlLoader(LoaderBase):
    def __init__(self, path: Path, time_field: str = "Time"):
        super().__init__(path)
        self.time_field = time_field

    def load(self, keys: Sequence[str]) -> JsonlSeries:
        if not self.source.exists():
            return JsonlSeries(times=np.array([]), values={k: np.array([]) for k in keys})

        times: List[float] = []
        values: Dict[str, List[float]] = {k: [] for k in keys}

        with self.source.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                try:
                    times.append(timestamp_to_seconds(str(payload[self.time_field])))
                except Exception:
                    continue

                for key in keys:
                    value = _extract_nested_value(payload, key)
                    values[key].append(value)

        aligned = {}
        for k, v in values.items():
            try:
                aligned[k] = np.array(v)
            except Exception:
                aligned[k] = np.array(v, dtype=object)
        return JsonlSeries(times=np.array(times), values=aligned)
