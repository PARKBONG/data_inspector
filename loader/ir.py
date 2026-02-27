import struct
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .base import LoaderBase


@dataclass
class IRFrame:
    timestamp: float
    data: np.ndarray


class IRRawLoader(LoaderBase):
    def __init__(self, path: Path, temp_range: Tuple[float, float]):
        super().__init__(path)
        self.temp_range = temp_range
        self.width = 0
        self.height = 0
        self.frame_stride = 0
        self.frame_count = 0
        self.timestamps: np.ndarray = np.array([])

        if self.source.exists():
            self._load_meta()

    def _load_meta(self):
        size = self.source.stat().st_size
        with self.source.open("rb") as handle:
            header = handle.read(8)
            self.width, self.height = struct.unpack("<II", header)
        self.frame_stride = 4 + self.width * self.height * 2
        self.frame_count = max(0, (size - 8) // self.frame_stride)
        self.timestamps = self._load_timestamps()

    def _load_timestamps(self) -> np.ndarray:
        if self.frame_count == 0:
            return np.array([])

        stamps: List[float] = []
        with self.source.open("rb") as handle:
            handle.seek(8)
            for _ in range(self.frame_count):
                data = handle.read(4)
                if not data:
                    break
                ms = struct.unpack("<i", data)[0]
                stamps.append(ms / 1000.0)
                handle.seek(self.width * self.height * 2, 1)
        return np.array(stamps)

    def load(self):
        return self

    def get_nearest_frame(self, timestamp: float) -> Optional[IRFrame]:
        if self.frame_count == 0 or len(self.timestamps) == 0:
            return None
        idx = int(np.argmin(np.abs(self.timestamps - timestamp)))
        return self.get_frame(idx)

    def get_frame(self, index: int) -> Optional[IRFrame]:
        if index < 0 or index >= self.frame_count:
            return None

        with self.source.open("rb") as handle:
            handle.seek(8 + index * self.frame_stride)
            ms = struct.unpack("<i", handle.read(4))[0]
            raw_pixels = handle.read(self.width * self.height * 2)

        frame = np.frombuffer(raw_pixels, dtype="<u2").reshape(self.height, self.width)
        return IRFrame(timestamp=ms / 1000.0, data=frame)

    def frame_values(self, timestamp: float) -> Optional[np.ndarray]:
        frame = self.get_nearest_frame(timestamp)
        if not frame:
            return None
        vmin, vmax = self.temp_range
        clipped = np.clip(frame.data, vmin, vmax)
        return clipped.astype(float)
