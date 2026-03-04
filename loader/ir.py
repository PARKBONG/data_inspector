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


@dataclass
class IRFileMeta:
    path: Path
    width: int
    height: int
    frame_stride: int
    frame_count: int


class IRRawLoader(LoaderBase):
    def __init__(self, path: Path, temp_range: Tuple[float, float]):
        super().__init__(path)
        self.temp_range = temp_range
        self.width = 0
        self.height = 0
        self.frame_count = 0
        self.timestamps: np.ndarray = np.array([])
        self.file_metas: List[IRFileMeta] = []

        if self.source.exists():
            self._load_meta()

    def _load_meta(self):
        sources = []
        if self.source.is_dir():
            sources = list(self.source.glob("*.raw"))
            sources.sort(key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
        else:
            sources = [self.source]

        all_stamps: List[float] = []

        for src in sources:
            size = src.stat().st_size
            if size < 8:
                continue

            with src.open("rb") as handle:
                header = handle.read(8)
                w, h = struct.unpack("<II", header)

            stride = 4 + w * h * 2
            count = max(0, (size - 8) // stride)
            if count == 0:
                continue

            if self.width == 0:
                self.width, self.height = w, h

            # Load timestamps for this specific file
            stamps = []
            with src.open("rb") as handle:
                handle.seek(8)
                for _ in range(count):
                    data = handle.read(4)
                    if not data:
                        break
                    ms = struct.unpack("<i", data)[0]
                    stamps.append(ms / 1000.0)
                    handle.seek(w * h * 2, 1)

            self.file_metas.append(IRFileMeta(src, w, h, stride, count))
            all_stamps.extend(stamps)

        self.timestamps = np.array(all_stamps)
        self.frame_count = len(self.timestamps)

    def load(self, start_time: float = None, end_time: float = None):
        if start_time is not None or end_time is not None:
            mask = np.ones(len(self.timestamps), dtype=bool)
            if start_time is not None: mask &= (self.timestamps >= start_time)
            if end_time is not None: mask &= (self.timestamps <= end_time)
            self.timestamps = self.timestamps[mask]
            self.frame_count = len(self.timestamps)
            # Note: mapping back to raw file offsets remains correct through timestamps filter
        return self

    def get_nearest_frame(self, timestamp: float) -> Optional[IRFrame]:
        if self.frame_count == 0 or len(self.timestamps) == 0:
            return None
        idx = int(np.argmin(np.abs(self.timestamps - timestamp)))
        return self.get_frame(idx)

    def get_frame(self, index: int) -> Optional[IRFrame]:
        if index < 0 or index >= self.frame_count:
            return None

        # Find which file contains this frame index
        curr_idx = index
        target_meta = None
        for meta in self.file_metas:
            if curr_idx < meta.frame_count:
                target_meta = meta
                break
            curr_idx -= meta.frame_count

        if not target_meta:
            return None

        with target_meta.path.open("rb") as handle:
            handle.seek(8 + curr_idx * target_meta.frame_stride)
            ms = struct.unpack("<i", handle.read(4))[0]
            raw_pixels = handle.read(target_meta.width * target_meta.height * 2)

        frame = np.frombuffer(raw_pixels, dtype="<u2").reshape(target_meta.height, target_meta.width)
        return IRFrame(timestamp=ms / 1000.0, data=frame)

    def frame_values(self, timestamp: float) -> Optional[np.ndarray]:
        frame = self.get_nearest_frame(timestamp)
        if not frame:
            return None
        vmin, vmax = self.temp_range
        clipped = np.clip(frame.data, vmin, vmax)
        return clipped.astype(float)
