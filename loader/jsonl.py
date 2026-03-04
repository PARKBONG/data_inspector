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

    def load(self, keys: Sequence[str], start_time: float = None, end_time: float = None) -> JsonlSeries:
        # 1. Check Cache
        cache_dir = self.source.parent / ".cache"
        cache_file = cache_dir / f"{self.source.name}.v2.npz"
        
        # Simple heuristic: if any jsonl is newer than cache, reload
        should_reload = True
        if cache_file.exists():
            cache_mtime = cache_file.stat().st_mtime
            latest_src_mtime = 0
            src_files = list(self.source.glob("*.jsonl")) if self.source.is_dir() else [self.source]
            for s in src_files:
                latest_src_mtime = max(latest_src_mtime, s.stat().st_mtime)
            if latest_src_mtime < cache_mtime:
                should_reload = False

        if not should_reload:
            try:
                data = np.load(cache_file, allow_pickle=True)
                times = data["_times"]
                # Only extract requested keys that exist in cache
                values = {k: data[k] for k in keys if k in data}
                # Check if we got all keys. If not, maybe keys changed, reload.
                if all(k in values for k in keys):
                    return self._filter_series(JsonlSeries(times=times, values=values), start_time, end_time)
            except Exception:
                pass # Fallback to slow load

        # 2. Slow Load
        if not self.source.exists():
            return JsonlSeries(times=np.array([]), values={k: np.array([]) for k in keys})

        sources = []
        if self.source.is_dir():
            sources = list(self.source.glob("*.jsonl"))
            sources.sort(key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
        else:
            sources = [self.source]

        all_times: List[float] = []
        temp_values: Dict[str, List[Any]] = {}

        for src in sources:
            with src.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line: continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError: continue
                    
                    t = timestamp_to_seconds(str(payload.get(self.time_field, 0)))
                    all_times.append(t)
                    
                    # Extract specifically requested keys (handling nested paths)
                    for k in keys:
                        v = _extract_nested_value(payload, k)
                        if k not in temp_values:
                            # Initialize with NaNs or Nones if we just discovered this key
                            # but we had previous lines without it.
                            if v is not None and isinstance(v, str):
                                temp_values[k] = [None] * (len(all_times) - 1)
                            else:
                                temp_values[k] = [np.nan] * (len(all_times) - 1)
                        
                        temp_values[k].append(v)

                    # Backfill any key that might have been in temp_values but wasn't in keys
                    # (though in this new logic, temp_values usually only contains 'keys')
                    for k in temp_values:
                        if len(temp_values[k]) < len(all_times):
                            placeholder = np.nan if not isinstance(temp_values[k][0], str) else None
                            temp_values[k].append(placeholder)

        # 3. Save Cache
        if not cache_dir.exists(): cache_dir.mkdir(parents=True, exist_ok=True)
        save_dict = {"_times": np.array(all_times)}
        for k, v in temp_values.items():
            # Try to force float conversion for numeric series
            try:
                arr = np.array(v)
                if arr.dtype == object:
                    # Try to see if it's mostly numbers/NaNs
                    save_dict[k] = arr.astype(float)
                else:
                    save_dict[k] = arr
            except Exception:
                save_dict[k] = np.array(v, dtype=object)
        
        try:
            np.savez(cache_file, **save_dict)
        except Exception:
            pass

        # 4. Filter and Return
        result_values = {}
        for k in keys:
            if k in save_dict:
                result_values[k] = save_dict[k]
            else:
                # Fill missing requested key with NaNs
                result_values[k] = np.full(len(all_times), np.nan)
                
        series = JsonlSeries(times=save_dict["_times"], values=result_values)
        return self._filter_series(series, start_time, end_time)

    def _filter_series(self, series: JsonlSeries, start: float, end: float) -> JsonlSeries:
        if start is None and end is None:
            return series
        
        times = series.times
        if len(times) == 0:
            return series
            
        mask = np.ones(len(times), dtype=bool)
        if start is not None:
            mask &= (times >= start)
        if end is not None:
            mask &= (times <= end)
            
        return JsonlSeries(
            times=times[mask],
            values={k: v[mask] for k, v in series.values.items()}
        )
